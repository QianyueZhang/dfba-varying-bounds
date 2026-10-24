"""Post-audit, frozen-tolerance diagnostic of every failed prefix root.

This diagnostic preserves the completed deployment audit and prescribed roots.
Every arm chains its own five intervals from the common physical initial state.
Run --declare, then --execute; neither mode overwrites an existing output.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import engine
import independent_amount_model as independent

PROTOCOL = ROOT / 'audit/prefix_diagnostic_protocol.json'
RESULTS = ROOT / 'audit/prefix_diagnostic_results.json'
INITIAL = [.5, .0005, .0005, 10.]
INPUTS = [
    'audit/deployment_results.json', 'audit/deployment_protocol.json',
    'audit/independent_amount_model.py', 'audit/diagnose_prefixes.py',
    'scripts/engine.py', 'vendor/prior_kernel.py',
    'data/flux_table_eps1e-06.json', 'results/roots.json',
]
ARMS = {
    'independent_original': {'rtol': 2e-13, 'atol': 2e-14, 'maxstep_h': .025},
    'independent_refined': {'rtol': 2e-14, 'atol': 2e-15, 'maxstep_h': .005},
    'engine_finer': {'rtol': 2e-13, 'atol': 2e-15, 'maxstep_h': .01},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_complete():
    deployment = json.loads((ROOT / 'audit/deployment_results.json').read_text())
    if deployment.get('complete') is not True:
        raise RuntimeError('Deployment is incomplete; declaration/execution forbidden')
    roots = json.loads((ROOT / 'results/roots.json').read_text())
    return deployment, {r['id']: r for r in roots}


def selected_prefixes(deployment):
    transitions = {t['id']: t for t in deployment['transitions']}
    failed_ids = {f['transition_id'] for f in deployment['failures'] if 'transition_id' in f}
    selected = []
    for prefix in deployment['prefix_replays']:
        reasons = []
        for stage in prefix['stages']:
            ident = stage['transition_id']
            transition = transitions[ident]
            if stage.get('saved_prefix_comparison', {}).get('passed') is not True:
                reasons.append({'stage': stage['stage'], 'kind': 'failed_saved_prefix', 'transition_id': ident})
            if (transition.get('comparison', {}).get('passed') is not True
                    or transition.get('complete') is not True or ident in failed_ids):
                reasons.append({'stage': stage['stage'], 'kind': 'failed_matched_prefix_transition', 'transition_id': ident})
        if reasons:
            selected.append({'root': prefix['root'], 'selection_reasons': reasons})
    return selected


def runtime():
    return {
        'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(),
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'numba']},
    }


def declare():
    if PROTOCOL.exists() or RESULTS.exists():
        raise FileExistsError('Diagnostic protocol/results already exist')
    deployment, roots = load_complete()
    selected = selected_prefixes(deployment)
    for item in selected:
        root = roots[item['root']]
        if len(root['feeds']) != 5 or len(root['knots']) != 6:
            raise ValueError('Every selected root must have five archived prefix intervals')
        if not np.allclose(independent.from_log(root['knots'][0]), INITIAL, rtol=0., atol=1e-18):
            raise ValueError('Unexpected initial root state')
    source_protocol = json.loads((ROOT / 'audit/deployment_protocol.json').read_text())
    protocol = {
        'declared_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'Declared after completed deployment and before any diagnostic integrations',
        'source_sha256': {p: digest(ROOT / p) for p in INPUTS},
        'runtime': runtime(),
        'deployment_original_passed': deployment['passed'],
        'deployment_original_failure_records': len(deployment['failures']),
        'selection_rule': 'Every root with any failed saved-prefix comparison or failed matched transition used in that prefix; no hand selection or duplicate root counting.',
        'selected_roots': selected,
        'initial_physical_amount_state': INITIAL,
        'state_order': ['volume_L', 'SC_gDW', 'KP_gDW', 'glucose_g'],
        'intervals': 5, 'interval_duration_h': 2., 'affinities': [1., 1.],
        'arms': ARMS,
        'scipy_rtol_floor': float(100 * np.finfo(float).eps),
        'scipy_rtol_note': 'Refined independent requested rtol=2e-14 is below the SciPy floor; effective rtol=100*float64 epsilon, approximately 2.220446049250313e-14. Record emitted warnings.',
        'thresholds_unchanged': source_protocol['thresholds'],
        'comparison_rule': 'For left minus right, amount allowance is abs_g + rel*abs(right amount), as in deployment. Compare volume and log ratio separately at each of five prefix endpoints.',
        'comparisons': [
            'engine_finer_vs_independent_refined',
            'independent_original_vs_independent_refined',
            'independent_original_vs_saved_engine',
            'independent_refined_vs_saved_engine',
            'engine_finer_vs_saved_engine',
            'independent_original_vs_archived_independent',
        ],
        'verification_rule': 'Original independent arm must recalculate every archived independent prefix endpoint with exact float64 array equality; also report unchanged-threshold comparisons and errors.',
        'chain_rule': 'All arms start from [0.5,0.0005,0.0005,10] and each uses only its own previous endpoint; no resetting to saved knots or prescribed roots.',
        'preservation_rule': 'Never overwrite or substitute roots, source code, main audit, failures, or thresholds. Diagnostic results neither erase nor convert the original audit failure into a pass.',
        'scope_limit': 'Post-audit convergence diagnostic of selected prefix roots only; does not certify integrators, controller trajectories, continuous uncertainty, or nonprefix failures.',
    }
    write_new(PROTOCOL, protocol)
    print(json.dumps({'declared': str(PROTOCOL), 'selected_roots': [x['root'] for x in selected]}, indent=2))


def compare(left, right, thresholds):
    left, right = np.asarray(left), np.asarray(right)
    error = np.abs(left - right)
    allowed = thresholds['amount_absolute_g'] + thresholds['amount_relative'] * np.abs(right[1:])
    z_error = abs(math.log(left[1]/left[2]) - math.log(right[1]/right[2]))
    passed = bool(np.all(error[1:] <= allowed)
                  and error[0] <= thresholds['volume_absolute_L']
                  and z_error <= thresholds['log_ratio_absolute'])
    return {
        'passed': passed, 'absolute_component_errors': error.tolist(),
        'amount_allowances': allowed.tolist(),
        'maximum_amount_error_over_allowance': float(np.max(error[1:] / allowed)),
        'log_ratio_error': float(z_error),
        'exact_array_equal': bool(np.array_equal(left, right)),
    }


def summary(comparisons):
    return {
        'comparisons': len(comparisons), 'passed': sum(c['passed'] for c in comparisons),
        'exact_array_equal': sum(c['exact_array_equal'] for c in comparisons),
        'maximum_component_absolute_error': np.max([c['absolute_component_errors'] for c in comparisons], axis=0).tolist() if comparisons else None,
        'maximum_amount_error_over_allowance': max((c['maximum_amount_error_over_allowance'] for c in comparisons), default=None),
        'maximum_log_ratio_error': max((c['log_ratio_error'] for c in comparisons), default=None),
    }


def execute():
    if RESULTS.exists():
        raise FileExistsError('Diagnostic results already exist')
    deployment, roots = load_complete()
    protocol = json.loads(PROTOCOL.read_text())
    protocol_hash = digest(PROTOCOL)
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Frozen source changed: ' + name)
    if selected_prefixes(deployment) != protocol['selected_roots']:
        raise RuntimeError('Root selection differs from declared protocol')
    if protocol['arms'] != ARMS:
        raise RuntimeError('Solver settings differ from declared protocol')
    thresholds = protocol['thresholds_unchanged']
    transitions = {t['id']: t for t in deployment['transitions']}
    prefixes = {p['root']: p for p in deployment['prefix_replays']}
    report = {
        'complete': False, 'protocol_sha256': protocol_hash,
        'source_sha256_before': protocol['source_sha256'],
        'runtime': runtime(), 'roots': [], 'exceptions': [],
        'original_audit_passed': deployment['passed'],
        'original_failure_records_retained': deployment['failures'],
        'interpretation': protocol['scope_limit'] + ' ' + protocol['preservation_rule'],
    }
    started = time.perf_counter()
    pools = {name: [] for name in protocol['comparisons']}
    for selection in protocol['selected_roots']:
        root = roots[selection['root']]
        prefix = prefixes[root['id']]
        row = {'root': root['id'], 'selection_reasons': selection['selection_reasons'],
               'family': root['family'], 'feeds': root['feeds'], 'arms': {}, 'knots': []}
        report['roots'].append(row)
        for arm, settings in protocol['arms'].items():
            arm_result = {'complete': False, 'requested_settings': settings,
                          'effective_rtol': max(settings['rtol'], protocol['scipy_rtol_floor']) if arm.startswith('independent') else settings['rtol'],
                          'initial_amount_state': INITIAL, 'stages': []}
            row['arms'][arm] = arm_result
            state = np.asarray(INITIAL, float).copy()
            model = independent.AmountReactor(root['family']) if arm.startswith('independent') else None
            args = engine.arguments(root['family']) if model is None else None
            if model is not None:
                independent.RTOL, independent.ATOL, independent.MAX_STEP = settings['rtol'], settings['atol'], settings['maxstep_h']
            for stage, feed in enumerate(root['feeds']):
                try:
                    tick = time.perf_counter()
                    with warnings.catch_warnings(record=True) as recorded:
                        warnings.simplefilter('always')
                        if model is not None:
                            out = model.interval(state, float(feed), np.array([1., 1.]))
                            endpoint, minimum = out['amount_state'], out['min_S']
                            details = {'rhs_calls': int(out['rhs_calls']), 'max_S': float(out['max_S'])}
                        else:
                            out, minimum, accepted, rejected = engine.flow(
                                independent.to_log(state), float(feed), np.array([1., 1.]), args,
                                settings['rtol'], settings['atol'], settings['maxstep_h'])
                            endpoint = independent.from_log(out)
                            details = {'accepted_steps': int(accepted), 'rejected_steps': int(rejected)}
                    endpoint = np.asarray(endpoint)
                    physical = bool(np.all(np.isfinite(endpoint)) and min(endpoint[:3]) > 0
                                    and minimum >= thresholds['physical_substrate_min_g_L']
                                    and endpoint[0] <= thresholds['volume_cap_L']
                                    and 0 <= feed <= thresholds['feed_max_L_h'])
                    arm_result['stages'].append({
                        'stage': stage, 'input_amount_state': state.tolist(), 'amount_state': endpoint.tolist(),
                        'min_S': float(minimum), 'physical_valid': physical,
                        'analytic_volume_error_L': abs(float(endpoint[0]) - (float(state[0]) + 2.*feed)),
                        'seconds': time.perf_counter() - tick,
                        'warnings': [str(w.message) for w in recorded], **details})
                    state = endpoint.copy()
                except Exception as error:
                    report['exceptions'].append({'root': root['id'], 'arm': arm, 'stage': stage,
                                                 'type': type(error).__name__, 'message': str(error)})
                    break
            arm_result['complete'] = len(arm_result['stages']) == 5
        for stage in range(5):
            if any(len(a['stages']) <= stage for a in row['arms'].values()):
                break
            states = {a: x['stages'][stage]['amount_state'] for a, x in row['arms'].items()}
            states['saved_engine'] = independent.from_log(root['knots'][stage+1]).tolist()
            ident = prefix['stages'][stage]['transition_id']
            states['archived_independent'] = transitions[ident]['independent']['amount_state']
            comparisons = {}
            for name in protocol['comparisons']:
                left, right = name.split('_vs_')
                comparison = compare(states[left], states[right], thresholds)
                comparisons[name] = comparison
                pools[name].append(comparison)
            row['knots'].append({'stage': stage, 'time_h': 2.*(stage+1), 'archived_transition_id': ident,
                                 'saved_engine_amount_state': states['saved_engine'],
                                 'archived_independent_amount_state': states['archived_independent'],
                                 'comparisons': comparisons})
    report['summary'] = {name: summary(values) for name, values in pools.items()}
    report['original_matched_exactly'] = all(c['exact_array_equal'] for c in pools['independent_original_vs_archived_independent']) and len(pools['independent_original_vs_archived_independent']) == 5*len(protocol['selected_roots'])
    report['all_arms_physically_valid'] = all(s['physical_valid'] for r in report['roots'] for a in r['arms'].values() for s in a['stages']) and not report['exceptions']
    report['source_sha256_after'] = {p: digest(ROOT / p) for p in INPUTS}
    report['frozen_inputs_unchanged'] = report['source_sha256_before'] == report['source_sha256_after'] and digest(PROTOCOL) == protocol_hash
    report['complete'] = not report['exceptions'] and all(a['complete'] for r in report['roots'] for a in r['arms'].values())
    report['diagnostic_valid'] = report['complete'] and report['original_matched_exactly'] and report['frozen_inputs_unchanged']
    report['elapsed_seconds'] = time.perf_counter() - started
    write_new(RESULTS, report)
    print(json.dumps({k: report[k] for k in ['complete', 'diagnostic_valid', 'original_matched_exactly', 'all_arms_physically_valid', 'frozen_inputs_unchanged', 'summary', 'elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--declare', action='store_true')
    group.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    declare() if options.declare else execute()
