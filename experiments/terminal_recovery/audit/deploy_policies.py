"""Independent causal deployment on every fixed-target primary and OOD path.

Declare first; execute only after the primary design has all 108 rows. Controller
decisions never receive a disturbance index or the independently used plant
parameters. Cache keys use exact floating-point states, without rounding.
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
PROTOCOL = HERE / 'deployment_protocol.json'
RESULTS = HERE / 'deployment_results.json'
METHODS = ['nominal_replan', 'open_grid', 'open_replan', 'recovery_economic', 'recovery_filter']
TARGETS = [.5, .75, 1., 1.25, 1.5, 2.]
K = [[1., 1.], [.975, 1.025], [1.025, .975]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def finite(value):
    """Represent infinities explicitly as null, with status fields explaining them."""
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if hasattr(value, 'tolist'):
        return finite(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def declare():
    if PROTOCOL.exists() or RESULTS.exists():
        raise RuntimeError('Deployment already declared or run; do not overwrite')
    design_lock = json.loads((HERE / 'design_lock.json').read_text())
    for path, expected in design_lock.items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Design input changed: ' + path)
    paths = list(design_lock) + ['audit/design_lock.json', 'audit/deploy_policies.py',
        'EXPLORATORY_TARGET_SET_PROTOCOL.md', 'scripts/target_sets.py', 'audit/target_set_lock.json']
    witness_count = len(json.loads((ROOT / 'results/target_set_witnesses.json').read_text()))
    protocol = dict(
        declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='Declared before independent deployment execution',
        source_sha256={p: sha(ROOT / p) for p in paths},
        primary=dict(roots=18, targets=TARGETS, problems=108, methods=METHODS,
                     affinity_set=K, paths=list(itertools.product(range(3), repeat=2)), episodes=4860),
        source_files=['results/roots.json', 'results/primary_design.json', 'results/target_sets.json', 'results/target_set_witnesses.json'],
        source_binding='Require exactly 18 declared roots and all 108 unique fixed-target rows, then bind complete-file hashes before execution. Reject partial inputs.',
        root_rule='Engine-generated roots are prescribed primary initial states. Independently replay all 18 nominal five-interval prefixes separately; do not substitute replay endpoints into primary roots.',
        first_action='Execute the saved per-method first feed and retain its plan/status/proposal fields.',
        second_action='Fixed open_grid commits saved plan[1]. Other methods replan from independent actual state using the frozen one-step controller; no hidden affinity, inhibition, scenario identity or future state supplied.',
        nominal_guard='Nominal composition and biomass objective with all-scenario physical guard, as frozen in the main protocol.',
        safe_label_rule='Keep decision diagnostic, robust state recoverability, selected action robustness, achieved terminal bands and optimal-state J distinct. Stored J is not necessarily achieved by selected economic/filter policy.',
        cache='Exact keys: family, inhibition, physical state bytes, feed, affinity for independent transitions; family and observed log-state bytes for controller maps. No state rounding. Targets/methods share identical transitions/maps.',
        independent=dict(class_name='AmountReactor', rtol=2e-13, atol=2e-14, maxstep_h=.025),
        engine=dict(rtol=2e-11, atol=2e-13, maxstep_h=.05),
        thresholds=dict(amount_absolute_g=2e-7, amount_relative=2e-8, log_ratio_absolute=1e-7,
                        volume_absolute_L=2e-13, physical_substrate_min_g_L=-2e-7,
                        volume_cap_L=1.8, feed_max_L_h=.2, inner=.00748, outer=.0075,
                        actual=math.log(1.01), comparison_tolerance=1e-10),
        matching_rule='Every distinct independent interval is compared with engine.flow from the SAME actual input state, feed, affinity and plant inhibition. This matched audit use of true parameters never enters a controller decision. Prefix cumulative endpoints are additionally compared with saved knots.',
        classification_rule='Inner uses inner+1e-10; outer and actual use their stated bands without added comparison tolerance. All require physical validity and complete integration. Operational success additionally requires no controller exception. Compare final matched-engine and independent labels in each band; retain disagreements as hard audit failures.',
        failure_rule='Retain all assigned episodes, failed evaluations, numerical discrepancies and incomplete paths. No silent fallback on a controller exception; mark path incomplete/unsuccessful and continue remaining assigned paths. Never loosen thresholds.',
        ood=dict(selection='All six uniform-history roots and six fixed targets, all five methods', episodes=720,
                 stresses=[
                     dict(id='persistent_minus', kappas=[[.92, 1.08], [.92, 1.08]], inhibition=None, observation_offset=None),
                     dict(id='persistent_plus', kappas=[[1.08, .92], [1.08, .92]], inhibition=None, observation_offset=None),
                     dict(id='omitted_inhibition', kappas=[[1., 1.], [1., 1.]], inhibition=[180., 260.], observation_offset=None),
                     dict(id='biased_second_observation', kappas=[[1., 1.], [1., 1.]], inhibition=None,
                          observation_offset=dict(logM1=.0025, logM2=-.0025, substrate_g_L=.025)),
                 ],
                 interpretation='Secondary, outside the primary finite-game assumptions. Fixed deterministic stresses, not biological replicates or new confidence-region evidence.'),
        exploratory=dict(selection='Log midpoint of EVERY connected feedback target component from target_sets.json; verify one-to-one component coverage before execution',
                         methods=['recovery_economic', 'recovery_filter'], witnesses=witness_count, episodes=witness_count*18,
                         interpretation='Exploratory constructive-policy consistency checks on selected feasible targets. Never merge with the fixed-target primary denominator or present as an unbiased success rate.'),
        timing='Report unique independent calls, cache reuse, map calls, decision calls, matched engine calls and wall time. Offline cached audit timing is not complete online-controller latency.',
    )
    save(PROTOCOL, protocol)
    print(json.dumps(dict(declared=True, primary=4860, ood=720, exploratory=witness_count*18, prefixes=18, numerical_execution=False)))


def execute():
    if not PROTOCOL.exists():
        raise RuntimeError('Declare deployment protocol first')
    if RESULTS.exists():
        raise RuntimeError('Deployment result already exists; do not overwrite')
    protocol = json.loads(PROTOCOL.read_text())
    for path, expected in protocol['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Declared source changed: ' + path)
    roots = json.loads((ROOT / 'results/roots.json').read_text())
    rows = json.loads((ROOT / 'results/primary_design.json').read_text())
    expected_roots = {f'{family}_v{v:.1f}_{timing}' for family in ('analytic', 'table')
                      for v in (.9, 1.3, 1.7) for timing in ('early', 'late', 'uniform')}
    if len(roots) != 18 or {r['id'] for r in roots} != expected_roots:
        raise RuntimeError('Missing, duplicate or unexpected root states')
    expected_rows = {(root, target) for root in expected_roots for target in TARGETS}
    if len(rows) != 108 or {(r['root'], r['target']) for r in rows} != expected_rows:
        raise RuntimeError('Primary design is partial or does not contain the declared 108 rows')
    for row in rows:
        if set(row['choices']) != set(METHODS):
            raise RuntimeError('Method choices incomplete')
    target_sets = json.loads((ROOT / 'results/target_sets.json').read_text())
    witnesses = json.loads((ROOT / 'results/target_set_witnesses.json').read_text())
    set_lookup = {row['root']: row for row in target_sets}
    expected_components = {(row['root'], index) for row in target_sets for index in range(len(row['feedback']))}
    if len(witnesses) != protocol['exploratory']['witnesses'] or {(w['root'], w['component']) for w in witnesses} != expected_components:
        raise RuntimeError('Exploratory witness coverage differs from every connected component')
    for witness in witnesses:
        interval = set_lookup[witness['root']]['feedback'][witness['component']]
        if witness['log_interval'] != interval or abs(math.log(witness['target'])-sum(interval)/2) > 1e-14:
            raise RuntimeError('Exploratory witness is not the declared component midpoint')
        if set(witness['choices']) != set(protocol['exploratory']['methods']):
            raise RuntimeError('Exploratory witness methods differ')
    source_hashes = {p: sha(ROOT / p) for p in protocol['source_files']}
    row_lookup = {(r['root'], r['target']): r for r in rows}

    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(HERE))
    import numpy as np
    import scipy
    import numba
    import engine
    import policies
    import independent_amount_model as independent

    t = protocol['thresholds']
    if not np.array_equal(engine.K, np.asarray(K)) or not np.array_equal(engine.GRID, np.linspace(0, .2, 21)):
        raise RuntimeError('Frozen action/affinity grid differs')
    if policies.INNER != t['inner'] or policies.TOL != t['comparison_tolerance']:
        raise RuntimeError('Frozen inner threshold differs')
    independent.RTOL = protocol['independent']['rtol']
    independent.ATOL = protocol['independent']['atol']
    independent.MAX_STEP = protocol['independent']['maxstep_h']
    report = dict(complete=False, passed=False, protocol_sha256=sha(PROTOCOL), protocol=protocol,
                  execution_input_sha256=source_hashes, prefix_replays=[], transitions=[], decisions=[],
                  primary_episodes=[], ood_episodes=[], exploratory_episodes=[], failures=[],
                  runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__))
    caches = dict(transitions={}, maps={}, decisions={}, model={}, arguments={})
    counts = collections.Counter()
    seconds = collections.Counter()
    started = time.perf_counter()

    def fail(code, **context):
        report['failures'].append(finite(dict(code=code, **context)))

    def args(family, inhibition=None):
        key = (family, None if inhibition is None else tuple(inhibition))
        if key not in caches['arguments']:
            caches['arguments'][key] = engine.arguments(family, inhibition)
        return caches['arguments'][key]

    def model(family, inhibition):
        key = (family, None if inhibition is None else tuple(inhibition))
        if key not in caches['model']:
            caches['model'][key] = independent.AmountReactor(family, inhibition)
        return caches['model'][key]

    def state_key(state):
        return np.asarray(state, dtype=np.float64).tobytes()

    def valid(state, minimum, feed):
        return bool(np.all(np.isfinite(state)) and state[0] > 0 and np.all(state[1:3] > 0)
                    and state[0] <= t['volume_cap_L'] and minimum >= t['physical_substrate_min_g_L']
                    and 0 <= feed <= t['feed_max_L_h'])

    def compare(left, right, context):
        a, b = np.asarray(left), np.asarray(right)
        error = np.abs(a-b)
        allowed = t['amount_absolute_g'] + t['amount_relative']*np.abs(b[1:])
        z_error = abs(math.log(a[1]/a[2])-math.log(b[1]/b[2]))
        passed = bool(np.all(error[1:] <= allowed) and error[0] <= t['volume_absolute_L'] and z_error <= t['log_ratio_absolute'])
        for i in np.flatnonzero(error[1:] > allowed):
            fail('physical_amount_discrepancy', **context, component=['SC_gDW', 'KP_gDW', 'glucose_g'][i],
                 error=float(error[i+1]), allowance=float(allowed[i]))
        if error[0] > t['volume_absolute_L']:
            fail('volume_discrepancy', **context, error_L=float(error[0]))
        if z_error > t['log_ratio_absolute']:
            fail('log_ratio_discrepancy', **context, error=z_error)
        return dict(passed=passed, absolute_component_errors=error.tolist(), amount_allowances=allowed.tolist(),
                    maximum_amount_error_over_allowance=float(np.max(error[1:]/allowed)), log_ratio_error=z_error)

    def transition(state, feed, kappa, family, inhibition, scope):
        counts['logical_interval_requests_' + scope] += 1
        key = (family, None if inhibition is None else tuple(inhibition), state_key(state), float(feed), tuple(kappa))
        if key in caches['transitions']:
            counts['transition_cache_hits'] += 1
            result = report['transitions'][caches['transitions'][key]]
            result['logical_uses'][scope] = result['logical_uses'].get(scope, 0)+1
            return result
        ident = len(report['transitions'])
        result = dict(id=ident, family=family, inhibition=inhibition, input_amount_state=np.asarray(state).tolist(),
                      feed=float(feed), kappa=kappa, logical_uses={scope: 1}, complete=False)
        caches['transitions'][key] = ident
        report['transitions'].append(result)
        clock = time.perf_counter()
        try:
            out = model(family, inhibition).interval(state, float(feed), kappa)
            duration = time.perf_counter()-clock
            seconds['independent_intervals'] += duration
            counts['independent_interval_calls'] += 1
            result['independent'] = dict(amount_state=out['amount_state'].tolist(), min_S=out['min_S'],
                max_S=out['max_S'], rhs_calls=out['rhs_calls'], seconds=duration,
                physical_valid=valid(out['amount_state'], out['min_S'], feed))
            if not result['independent']['physical_valid']:
                fail('independent_physical_invalid', transition_id=ident, scope=scope)
            volume_error = abs(float(out['amount_state'][0])-(float(state[0])+2.*feed))
            if volume_error > t['volume_absolute_L']:
                fail('independent_analytic_volume', transition_id=ident, error_L=volume_error)
        except Exception as error:
            seconds['independent_intervals'] += time.perf_counter()-clock
            counts['independent_interval_exceptions'] += 1
            result['independent_exception'] = dict(type=type(error).__name__, message=str(error))
            fail('independent_integration_exception', transition_id=ident, scope=scope, **result['independent_exception'])
        clock = time.perf_counter()
        try:
            endpoint, minimum, accepted, rejected = engine.flow(independent.to_log(state), float(feed), np.asarray(kappa),
                args(family, inhibition), protocol['engine']['rtol'], protocol['engine']['atol'], protocol['engine']['maxstep_h'])
            duration = time.perf_counter()-clock
            seconds['matched_engine_intervals'] += duration
            counts['matched_engine_interval_calls'] += 1
            endpoint = independent.from_log(endpoint)
            result['engine'] = dict(amount_state=endpoint.tolist(), min_S=float(minimum), accepted_steps=int(accepted),
                rejected_steps=int(rejected), seconds=duration, physical_valid=valid(endpoint, minimum, feed))
            if not result['engine']['physical_valid']:
                fail('matched_engine_physical_invalid', transition_id=ident, scope=scope)
        except Exception as error:
            seconds['matched_engine_intervals'] += time.perf_counter()-clock
            counts['matched_engine_interval_exceptions'] += 1
            result['engine_exception'] = dict(type=type(error).__name__, message=str(error))
            fail('matched_engine_integration_exception', transition_id=ident, scope=scope, **result['engine_exception'])
        if 'independent' in result and 'engine' in result:
            result['comparison'] = compare(result['engine']['amount_state'], result['independent']['amount_state'],
                                           dict(transition_id=ident, scope='matched_interval'))
            if result['independent']['physical_valid'] != result['engine']['physical_valid']:
                fail('matched_physical_label_disagreement', transition_id=ident)
        result['complete'] = 'independent' in result
        return result

    def get_map(observed, family):
        key = (family, state_key(observed))
        counts['controller_map_requests'] += 1
        if key in caches['maps']:
            counts['controller_map_cache_hits'] += 1
            return caches['maps'][key]
        clock = time.perf_counter()
        value = engine.one_step(observed, args(family))
        seconds['controller_one_step_maps'] += time.perf_counter()-clock
        counts['controller_one_step_maps'] += 1
        counts['controller_map_transitions'] += int(value['transitions'])
        caches['maps'][key] = value
        return value

    def second_decision(observed, family, target, method, saved_choice):
        committed = float(saved_choice['plan'][1]) if method == 'open_grid' else None
        key = (family, state_key(observed), float(target), method, committed)
        counts['second_decision_requests'] += 1
        if key in caches['decisions']:
            counts['second_decision_cache_hits'] += 1
            return report['decisions'][caches['decisions'][key]]
        ident = len(report['decisions'])
        clock = time.perf_counter()
        mapping = get_map(observed, family)
        robust = policies.one_values(mapping, target)
        nominal = policies.one_values(mapping, target, nominal=True)
        if method == 'open_grid':
            choice = dict(feed=committed, status='committed_open_plan', plan=saved_choice['plan'],
                          proposal=None, J=None, recoverable=None, index=None)
        else:
            # The controller receives only observed state, family model, target,
            # remaining horizon and method. No true plant parameter is passed.
            choice = policies.decision(observed, args(family), target, 1, method, mapping)
        feed = float(choice['feed'])
        indices = np.flatnonzero(mapping['actions'] == feed)
        if len(indices):
            i = int(indices[0])
            selected = dict(robust_error=float(robust['errors'][i]), nominal_error=float(nominal['errors'][i]),
                robust_minimum_biomass=float(robust['biomass'][i]), nominal_biomass=float(nominal['biomass'][i]),
                physical_valid=bool(robust['valid'][i]), exact_action_in_map=True)
        else:
            endpoints, minima, _, _ = engine.batch(np.repeat(observed[None, :], 3, axis=0),
                                                   np.full(3, feed), np.asarray(K), args(family))
            counts['off_map_selected_action_diagnostic_transitions'] += 3
            z, biomass = policies.observable(endpoints)
            selected = dict(robust_error=float(np.max(np.abs(z-np.log(target)))), nominal_error=float(abs(z[0]-np.log(target))),
                robust_minimum_biomass=float(biomass.min()), nominal_biomass=float(biomass[0]),
                physical_valid=bool(minima.min() >= t['physical_substrate_min_g_L'] and endpoints[:, 0].max() <= t['volume_cap_L']),
                exact_action_in_map=False)
        selected['robust_inner_safe'] = selected['physical_valid'] and selected['robust_error'] <= t['inner']+t['comparison_tolerance']
        selected['nominal_inner_safe'] = selected['physical_valid'] and selected['nominal_error'] <= t['inner']+t['comparison_tolerance']
        duration = time.perf_counter()-clock
        seconds['second_decisions_including_map_misses'] += duration
        counts['second_decision_calls'] += 1
        if choice.get('proposal'):
            counts['proposal_probe_transitions'] += choice['proposal']['probe_transitions']
        result = finite(dict(id=ident, family=family, target=target, method=method, observed_log_state=observed,
            choice=choice, robust_state_recoverable=bool(len(robust['safe'])), nominal_state_recoverable=bool(len(nominal['safe'])),
            robust_safe_actions=mapping['actions'][robust['safe']], robust_optimal_state_error=robust['J'],
            selected_forecast=selected, seconds=duration))
        caches['decisions'][key] = ident
        report['decisions'].append(result)
        return result

    # Prefixes are independent chained trajectories from a common initial state.
    for root in roots:
        state = np.array([.5, .0005, .0005, 10.])
        points = []
        prefix = dict(root=root['id'], stages=points, complete=False)
        for stage, feed in enumerate(root['feeds']):
            tr = transition(state, float(feed), K[0], root['family'], None, 'prefix')
            point = dict(stage=stage, transition_id=tr['id'])
            points.append(point)
            if not tr['complete']:
                break
            state = np.asarray(tr['independent']['amount_state'])
            saved = independent.from_log(np.asarray(root['knots'][stage+1]))
            point['saved_prefix_comparison'] = compare(state, saved, dict(scope='saved_prefix', root=root['id'], stage=stage))
        prefix['complete'] = len(points) == 5 and 'saved_prefix_comparison' in points[-1]
        prefix['final_amount_state'] = state.tolist()
        report['prefix_replays'].append(prefix)
    print('prefix audit complete', len(report['prefix_replays']), 'failure records', len(report['failures']), flush=True)
    save(RESULTS, finite(report))

    def episode(root, target, method, path_id, kappas, inhibition=None, offset=None, scope='primary', row_override=None, component=None):
        row = row_lookup[(root['id'], target)] if row_override is None else row_override
        saved_choice = row['choices'][method]
        state = independent.from_log(np.asarray(root['state']))
        minimum = float(state[3]/state[0])
        stages = []
        item = dict(root=root['id'], family=root['family'], target=target, method=method, path=path_id,
                    plant_kappas=kappas, plant_inhibition=inhibition, second_observation_offset=offset,
                    saved_first_choice=finite(saved_choice), stages=stages, complete=False,
                    controller_exception=False, integration_exception=False)
        if component is not None:
            item['exploratory_component'] = component
        try:
            first_index = int(saved_choice['index'])
            item['robust_initial_state_recoverable'] = bool(len(row['values']['safe']))
            item['saved_first_action_recoverable_with_some_robust_continuation'] = first_index in row['values']['safe']
            for stage in range(2):
                if stage == 0:
                    feed = float(saved_choice['feed'])
                    stage_record = dict(stage=0, feed=feed, status=saved_choice['status'], source='saved_first_choice')
                else:
                    observed = independent.to_log(state)
                    if offset is not None:
                        observed[1] += offset['logM1']
                        observed[2] += offset['logM2']
                        observed[3] += offset['substrate_g_L']*observed[0]
                    try:
                        choice = second_decision(observed, root['family'], target, method, saved_choice)
                    except Exception as error:
                        item['controller_exception'] = True
                        item['exception'] = dict(type=type(error).__name__, message=str(error))
                        fail('controller_exception', scope=scope, root=root['id'], target=target, method=method, path=path_id, **item['exception'])
                        break
                    feed = choice['choice']['feed']
                    stage_record = dict(stage=1, feed=feed, status=choice['choice']['status'], decision_id=choice['id'],
                        selected_robust_inner_safe=choice['selected_forecast']['robust_inner_safe'],
                        robust_state_recoverable=choice['robust_state_recoverable'])
                stages.append(stage_record)
                tr = transition(state, feed, kappas[stage], root['family'], inhibition, scope)
                stage_record['transition_id'] = tr['id']
                if not tr['complete']:
                    item['integration_exception'] = True
                    break
                state = np.asarray(tr['independent']['amount_state'])
                minimum = min(minimum, tr['independent']['min_S'])
                stage_record['independent_min_S'] = tr['independent']['min_S']
                stage_record['independent_physical_valid'] = tr['independent']['physical_valid']
            item['complete'] = len(stages) == 2 and 'independent_physical_valid' in stages[-1]
        except Exception as error:
            item['controller_exception'] = True
            item['exception'] = dict(type=type(error).__name__, message=str(error))
            fail('episode_exception', scope=scope, root=root['id'], target=target, method=method, path=path_id, **item['exception'])
        physical = bool(item['complete'] and all(s.get('independent_physical_valid', False) for s in stages))
        error = abs(math.log(state[1]/state[2])-math.log(target)) if item['complete'] else None
        labels = dict(inner=physical and error <= t['inner']+t['comparison_tolerance'],
                      outer=physical and error <= t['outer'], actual=physical and error <= t['actual']) if item['complete'] else dict(inner=False, outer=False, actual=False)
        item.update(final_amount_state=state.tolist(), min_S=minimum, physical_valid=physical if item['complete'] else None,
                    terminal_log_ratio_error=error, terminal_biomass_gDW=float(state[1]+state[2]) if item['complete'] else None,
                    terminal_band_success=labels,
                    operational_success=bool(labels['actual'] and not item['controller_exception'] and not item['integration_exception']))
        if item['complete']:
            final_transition = report['transitions'][stages[-1]['transition_id']]
            if 'engine' in final_transition:
                engine_state = final_transition['engine']['amount_state']
                engine_error = abs(math.log(engine_state[1]/engine_state[2])-math.log(target))
                engine_physical = all(s.get('independent_physical_valid', False) for s in stages[:-1]) and final_transition['engine']['physical_valid']
                engine_labels = dict(inner=engine_physical and engine_error <= t['inner']+t['comparison_tolerance'],
                                     outer=engine_physical and engine_error <= t['outer'], actual=engine_physical and engine_error <= t['actual'])
                item['matched_engine_terminal_band_success'] = engine_labels
                item['matched_engine_terminal_log_ratio_error'] = engine_error
                item['matched_terminal_classification_agrees'] = {band: engine_labels[band] == labels[band] for band in labels}
                for band in labels:
                    if engine_labels[band] != labels[band]:
                        fail('terminal_band_label_disagreement', scope=scope, root=root['id'], target=target,
                             method=method, path=path_id, band=band, transition_id=final_transition['id'])
        report[scope+'_episodes'].append(item)

    for root in roots:
        for target in TARGETS:
            for method in METHODS:
                for first, second in itertools.product(range(3), repeat=2):
                    episode(root, target, method, [first, second], [K[first], K[second]])
        save(RESULTS, finite(report))
        print('primary', root['id'], 'episodes', len(report['primary_episodes']), 'unique transitions', len(report['transitions']),
              'seconds', round(time.perf_counter()-started, 2), flush=True)
    for root in roots:
        if root['timing'] != 'uniform':
            continue
        for target in TARGETS:
            for method in METHODS:
                for stress in protocol['ood']['stresses']:
                    episode(root, target, method, stress['id'], stress['kappas'], stress['inhibition'], stress['observation_offset'], 'ood')
        save(RESULTS, finite(report))
        print('ood', root['id'], 'episodes', len(report['ood_episodes']), 'seconds', round(time.perf_counter()-started, 2), flush=True)
    for root in roots:
        for witness in witnesses:
            if witness['root'] != root['id']:
                continue
            logtarget = math.log(witness['target'])
            safe = [i for i, intervals in enumerate(set_lookup[root['id']]['by_first_action'])
                    if any(lo <= logtarget <= hi for lo, hi in intervals)]
            witness_row = dict(choices=witness['choices'], values=dict(safe=safe))
            for method in protocol['exploratory']['methods']:
                for first, second in itertools.product(range(3), repeat=2):
                    episode(root, witness['target'], method, [first, second], [K[first], K[second]],
                            scope='exploratory', row_override=witness_row, component=witness['component'])
        save(RESULTS, finite(report))
        print('exploratory', root['id'], 'episodes', len(report['exploratory_episodes']), 'seconds', round(time.perf_counter()-started, 2), flush=True)

    def summarize(episodes, scope):
        output = dict(assigned=len(episodes), complete=sum(r['complete'] for r in episodes), per_method={})
        output['matched_terminal_label_disagreements'] = {
            band: sum(r.get('matched_terminal_classification_agrees', {}).get(band) is False for r in episodes)
            for band in ('inner', 'outer', 'actual')}
        for method in (protocol['exploratory']['methods'] if scope == 'exploratory' else METHODS):
            selected = [r for r in episodes if r['method'] == method]
            grouped = collections.defaultdict(list)
            for row in selected:
                grouped[(row['root'], row['target'])].append(row)
            methods = dict(episodes=len(selected), complete=sum(r['complete'] for r in selected),
                          physical_valid=sum(r['physical_valid'] is True for r in selected),
                          operational_success=sum(r['operational_success'] for r in selected),
                          band_success={band: sum(r['terminal_band_success'][band] for r in selected) for band in ('inner', 'outer', 'actual')},
                          stage_status_counts=dict(collections.Counter(f"{s['stage']}:{s['status']}" for r in selected for s in r['stages'])),
                          root_target_worst_cases=[])
            for (root, target), group in grouped.items():
                complete = all(r['complete'] for r in group)
                methods['root_target_worst_cases'].append(dict(root=root, target=target, paths=len(group), complete=complete,
                    all_paths_band_success={band: all(r['terminal_band_success'][band] for r in group) for band in ('inner', 'outer', 'actual')},
                    maximum_log_ratio_error=max(r['terminal_log_ratio_error'] for r in group) if complete else None,
                    minimum_biomass_gDW=min(r['terminal_biomass_gDW'] for r in group) if complete else None))
            methods['all_paths_success_counts'] = {band: sum(r['all_paths_band_success'][band] for r in methods['root_target_worst_cases']) for band in ('inner', 'outer', 'actual')}
            if scope == 'ood':
                methods['by_stress'] = {stress['id']: dict(episodes=sum(r['path'] == stress['id'] for r in selected),
                    actual_success=sum(r['path'] == stress['id'] and r['terminal_band_success']['actual'] for r in selected)) for stress in protocol['ood']['stresses']}
            output['per_method'][method] = methods
        return output

    if len(report['primary_episodes']) != 4860 or len(report['ood_episodes']) != 720 or len(report['exploratory_episodes']) != protocol['exploratory']['episodes']:
        fail('assigned_denominator_mismatch', primary=len(report['primary_episodes']), ood=len(report['ood_episodes']), exploratory=len(report['exploratory_episodes']))
    for path, expected in {**protocol['source_sha256'], **source_hashes}.items():
        if sha(ROOT / path) != expected:
            fail('input_changed_during_execution', path=path)
    interval_comparisons = [r['comparison'] for r in report['transitions'] if 'comparison' in r]
    prefix_comparisons = [s['saved_prefix_comparison'] for p in report['prefix_replays'] for s in p['stages'] if 'saved_prefix_comparison' in s]

    def comparison_summary(values):
        return dict(comparisons=len(values), passed=sum(v['passed'] for v in values),
            maximum_component_absolute_error=np.max([v['absolute_component_errors'] for v in values], axis=0).tolist() if values else None,
            maximum_amount_error_over_allowance=max((v['maximum_amount_error_over_allowance'] for v in values), default=None),
            maximum_log_ratio_error=max((v['log_ratio_error'] for v in values), default=None))

    report['summary'] = dict(primary=summarize(report['primary_episodes'], 'primary'), ood=summarize(report['ood_episodes'], 'ood'),
        exploratory=summarize(report['exploratory_episodes'], 'exploratory'),
        prefixes=dict(assigned=18, complete=sum(p['complete'] for p in report['prefix_replays']), comparisons=comparison_summary(prefix_comparisons)),
        unique_matched_interval_comparisons=comparison_summary(interval_comparisons),
        unique_transitions=len(report['transitions']), unique_controller_maps=len(caches['maps']), unique_second_decisions=len(report['decisions']),
        counts=dict(counts), timing_seconds=dict(seconds), failed_records=len(report['failures']),
        failure_codes=dict(collections.Counter(f['code'] for f in report['failures'])))
    report['elapsed_seconds'] = time.perf_counter()-started
    report['complete'] = True
    report['passed'] = not report['failures']
    save(RESULTS, finite(report))
    print(json.dumps(dict(complete=True, passed=report['passed'], primary=len(report['primary_episodes']), ood=len(report['ood_episodes']),
                         exploratory=len(report['exploratory_episodes']), unique_transitions=len(report['transitions']), failures=len(report['failures']), seconds=report['elapsed_seconds']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--declare', action='store_true')
    group.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    declare() if options.declare else execute()


if __name__ == '__main__':
    main()
