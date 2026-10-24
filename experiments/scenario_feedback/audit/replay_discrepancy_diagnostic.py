"""Post hoc numerical-convergence diagnosis of frozen replay discrepancies.

Declare the protocol first with --declare; then use --execute. No controller
optimization is performed, and no original audit, threshold or result changes.
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
HERE = ROOT / 'audit'
PROTOCOL = HERE / 'replay_discrepancy_protocol.json'
OUTPUT = HERE / 'replay_discrepancy_results.json'
NOTE = HERE / 'REPLAY_DISCREPANCY_DIAGNOSTIC.md'
AUDIT = ROOT / 'results/locked/independent_amount_audit.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def declare():
    if PROTOCOL.exists():
        raise RuntimeError('Protocol already exists; do not overwrite a declared diagnostic')
    original = json.loads(AUDIT.read_text())
    failures = original['failures']
    selected = sorted({(f['context']['case'], f['context']['method']) for f in failures})
    inputs = ['results/locked/independent_amount_audit.json', 'LOCK.json']
    lock = json.loads((ROOT / 'LOCK.json').read_text())
    inputs.extend(lock['hashes'])
    inputs.extend('results/locked/' + case + '__' + method + '.json' for case, method in selected)
    inputs.append(str(Path(__file__).relative_to(ROOT)))
    for rel, expected in lock['hashes'].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError('Frozen input changed: ' + rel)
    settings = [
        dict(id='amount_audit', formulation='amount', rtol=2e-13, atol=2e-14, max_step_h=.025),
        dict(id='amount_refine_1', formulation='amount', rtol=5e-14, atol=5e-15, max_step_h=.0125),
        dict(id='amount_refine_2', formulation='amount', rtol=3e-14, atol=2e-15, max_step_h=.00625),
        dict(id='log_uncompiled_original', formulation='log', rtol=5e-12, atol=5e-14, max_step_h=.05),
        dict(id='log_uncompiled_refine_1', formulation='log', rtol=5e-14, atol=5e-15, max_step_h=.0125),
        dict(id='log_uncompiled_refine_2', formulation='log', rtol=3e-14, atol=2e-15, max_step_h=.00625),
    ]
    protocol = {
        'version': '1.0', 'declared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'post_hoc_diagnostic_declared_before_its_execution',
        'selection': 'Every unique episode in the completed original audit failure list; outcome-selected diagnostic, not new held-out evidence.',
        'original_failure_count': len(failures),
        'episodes': [{'case': case, 'method': method} for case, method in selected],
        'inputs_sha256': {rel: sha(ROOT / rel) for rel in sorted(set(inputs))},
        'method': 'DOP853', 'arms': settings,
        'replay': 'Chain saved executed feed and true-kappa intervals from the initial condition. Never reset to saved intermediate states; never optimize or change the executed policy.',
        'implementations': {
            'amount': 'Frozen independent AmountReactor.interval; numerical globals overridden in memory only.',
            'log': 'Frozen Reactor(fast=False).predict invokes its original uncompiled log-state RHS; only solve_ivp tolerances/max_step overridden in memory. Original analytic endpoint-volume correction retained.',
            'rhs_consistency': 'At up to 33 uniformly spaced solver-node indices per interval of the finest uncompiled log replay, compare independently assembled rates and physical-amount derivatives. This is a finite sample, not a proof.'},
        'tolerances': original['audit_protocol']['tolerances'],
        'physical_constraints': original['audit_protocol']['physical_constraints'],
        'comparisons': ['Every arm versus saved primary endpoints', 'Original independent arm versus archived audit endpoints',
                        'Each formulation refinement 1 versus refinement 2', 'Two finest formulations versus one another'],
        'interpretation_rule': 'Attribute a discrepancy as consistent with integration error only when both refinements and finest cross-formulation differences are materially smaller than the saved-to-refined discrepancy. Report all values even if nonmonotone; do not recategorize the original audit as passed.',
        'limitations': ['Six outcome-selected episodes only', 'Double precision and adaptive error estimators are not rigorous enclosures',
                       'Relative tolerances are above the SciPy roundoff floor, approximately 2.22e-14',
                       'Physical substrate minima are sampled at solver nodes, not globally certified',
                       'No online timing or controller-effectiveness inference'],
    }
    save(PROTOCOL, protocol)
    print(json.dumps({'protocol': str(PROTOCOL), 'episodes': len(selected), 'arms': len(settings), 'numerical_execution': False}, indent=2))


def execute():
    if not PROTOCOL.exists():
        raise RuntimeError('Declare the diagnostic protocol before execution')
    if OUTPUT.exists():
        raise RuntimeError('Diagnostic result already exists; do not overwrite it')
    protocol = json.loads(PROTOCOL.read_text())
    for rel, expected in protocol['inputs_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError('Declared input changed: ' + rel)
    import numpy as np
    import scipy
    from scipy.integrate import solve_ivp
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(HERE))
    import independent_amount_model as amount
    import base_model
    import model
    constraints = protocol['physical_constraints']
    tolerance = protocol['tolerances']
    original_audit = json.loads(AUDIT.read_text())
    original_rows = {(r['case'], r['method']): r for r in original_audit['episodes']}
    report = {'complete': False, 'protocol_sha256': sha(PROTOCOL), 'protocol': protocol,
              'original_audit_preserved': None, 'episodes': [],
              'runtime': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                          'platform': platform.platform(), 'uncompiled_log_rhs': True},
              'scope': 'Offline convergence diagnosis only. Original audit failure and original scientific outcomes retained.'}
    start = time.perf_counter()

    def difference(left, right):
        left = np.asarray(left, float)
        right = np.asarray(right, float)
        error = np.abs(left - right)
        allowed = tolerance['amount_atol_g'] + tolerance['amount_rtol'] * np.abs(right[:, 1:])
        log_error = np.abs(np.log(left[:, 1:3]) - np.log(right[:, 1:3]))
        exceeded = np.argwhere(error[:, 1:] > allowed)
        return {'max_physical_amount_absolute_error_g': float(error[:, 1:].max()),
                'component_max_absolute_error': error.max(axis=0).tolist(),
                'max_biomass_log_absolute_error': float(log_error.max()),
                'max_amount_error_over_frozen_tolerance': float((error[:, 1:] / allowed).max()),
                'unique_knot_component_exceedances': [
                    {'knot': int(k), 'component': ['SC_gDW', 'KP_gDW', 'glucose_g'][int(c)],
                     'absolute_error_g': float(error[k, c + 1]), 'allowed_error_g': float(allowed[k, c])}
                    for k, c in exceeded]}

    def classify(states, minima, row):
        endpoint = states[-1]
        physical = bool(all(0 <= s['feed'] <= constraints['feed_max_L_h'] for s in row['stages'])
                        and max(s[0] for s in states) <= constraints['volume_max_L']
                        and min(minima) >= constraints['substrate_min_g_L'])
        error = float(abs(np.log(endpoint[1] / endpoint[2]) - np.log(row['case']['target_ratio'])))
        controller_failed = any('controller_exception' in s['decision'] for s in row['stages'])
        terminal = bool(physical and error <= constraints['terminal_log_ratio_tolerance'])
        labels = dict(physical_valid=physical, terminal_feasible=terminal,
                      operational_success=bool(terminal and not controller_failed), controller_failure=controller_failed)
        return dict(labels, agrees_with_saved=all(labels[k] == row['result'][k] for k in labels),
                    log_ratio_error=error, terminal_boundary_distance=error-constraints['terminal_log_ratio_tolerance'],
                    biomass_amount_gDW=float(endpoint[1:3].sum()), minimum_substrate_g_L=float(min(minima)),
                    maximum_volume_L=float(max(s[0] for s in states)))

    for selected in protocol['episodes']:
        path = ROOT / 'results/locked' / (selected['case'] + '__' + selected['method'] + '.json')
        row = json.loads(path.read_text())
        case = row['case']
        saved = np.asarray([amount.from_log(s) for s in row['states']])
        archived = original_rows[(selected['case'], selected['method'])]
        archived_states = np.vstack([amount.initial_amounts(case['initial_ratio'], case['initial_glucose']),
                                     [s['amount_state'] for s in archived['trajectory']]])
        arms = {}
        for settings in protocol['arms']:
            y = amount.initial_amounts(case['initial_ratio'], case['initial_glucose'])
            log_state = model.initial_state(case['initial_ratio'], case['initial_glucose'])
            if settings['formulation'] == 'log':
                y = amount.from_log(log_state)
            states = [y.copy()]
            minimums = [float(y[3] / y[0])]
            stages = []
            check_rates = 0.
            check_rhs = 0.
            rhs_samples = 0
            plant_amount = amount.AmountReactor(case.get('family', 'table'), case.get('inhibition'))
            plant_log = model.Reactor(case.get('family', 'table'), case.get('inhibition'), fast=False)
            if plant_log.fast:
                raise RuntimeError('Uncompiled diagnostic unexpectedly activated native RHS')
            amount.RTOL, amount.ATOL, amount.MAX_STEP = settings['rtol'], settings['atol'], settings['max_step_h']

            def configured_integrator(fun, t_span, initial, **kwargs):
                if kwargs['method'] != 'DOP853':
                    raise RuntimeError('Unexpected original integrator method')
                kwargs.update(rtol=settings['rtol'], atol=settings['atol'], max_step=settings['max_step_h'])
                return solve_ivp(fun, t_span, initial, **kwargs)

            base_model.solve_ivp = configured_integrator
            for stage in row['stages']:
                feed = float(stage['feed'])
                kappa = stage['true_kappa']
                volume0 = float(y[0])
                if settings['formulation'] == 'amount':
                    out = plant_amount.interval(y, feed, kappa)
                    y = out['amount_state']
                else:
                    out = plant_log.predict(log_state, [feed], kappa, tight=True, record=True)
                    log_state = out['state'].copy()
                    log_state[0] = volume0 + amount.DT * feed
                    y = amount.from_log(log_state)
                    if settings['id'] == 'log_uncompiled_refine_2':
                        trajectory = np.asarray(out['trajectory'])
                        indices = np.unique(np.linspace(0, len(trajectory) - 1, min(33, len(trajectory))).astype(int))
                        for index in indices:
                            state = trajectory[index]
                            substrate = state[3] / state[0]
                            masses = np.exp(state[1:3])
                            independent_rates, _ = plant_amount.rates(substrate, kappa)
                            original_rates, _ = plant_log.rates(substrate, kappa)
                            independent_rhs = np.r_[independent_rates[:, 0] * masses, 100 * feed + .18 * independent_rates[:, 1] @ masses]
                            original_transformed_rhs = np.r_[original_rates[:, 0] * masses, 100 * feed + .18 * original_rates[:, 1] @ masses]
                            check_rates = max(check_rates, float(np.max(np.abs(independent_rates - original_rates))))
                            check_rhs = max(check_rhs, float(np.max(np.abs(independent_rhs - original_transformed_rhs))))
                            rhs_samples += 1
                minimums.append(out['min_S'])
                states.append(y.copy())
                stages.append(dict(stage=stage['stage'], feed=feed, kappa=kappa,
                                   amount_state=y.tolist(), min_S=out['min_S'], max_S=out['max_S'], rhs_calls=out['rhs_calls']))
            arms[settings['id']] = {'settings': settings, 'states': np.asarray(states).tolist(), 'stages': stages,
                                    'versus_saved': difference(states, saved), 'classification': classify(states, minimums, row)}
            if rhs_samples:
                arms[settings['id']]['sampled_formulation_consistency'] = {
                    'samples': rhs_samples, 'maximum_rate_absolute_difference': check_rates,
                    'maximum_transformed_rhs_absolute_difference': check_rhs}
            print(selected['case'], selected['method'], settings['id'],
                  'max_amount_error_g=', arms[settings['id']]['versus_saved']['max_physical_amount_absolute_error_g'], flush=True)
        comparisons = {
            'amount_audit_vs_archived_audit': difference(arms['amount_audit']['states'], archived_states),
            'amount_refinement_1_vs_2': difference(arms['amount_refine_1']['states'], arms['amount_refine_2']['states']),
            'log_refinement_1_vs_2': difference(arms['log_uncompiled_refine_1']['states'], arms['log_uncompiled_refine_2']['states']),
            'finest_amount_vs_finest_log': difference(arms['amount_refine_2']['states'], arms['log_uncompiled_refine_2']['states'])}
        report['episodes'].append(dict(selected, arms=arms, comparisons=comparisons))
        save(OUTPUT, report)
    for rel, expected in protocol['inputs_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError('Declared input changed during diagnostic: ' + rel)
    report['original_audit_preserved'] = True
    report['original_audit_hard_checks_passed'] = original_audit['hard_checks_passed']
    report['all_replay_classifications_agree_with_saved'] = all(a['classification']['agrees_with_saved'] for e in report['episodes'] for a in e['arms'].values())
    report['elapsed_offline_seconds'] = time.perf_counter() - start
    report['complete'] = True
    save(OUTPUT, report)
    lines = ['# Post hoc replay-discrepancy convergence diagnostic', '',
             'This diagnostic preserves the original failed independent audit and its frozen tolerances. It replays only the six unique flagged episodes with their saved executed feeds. It does not change or reoptimize the controllers.', '',
             'Protocol: `replay_discrepancy_protocol.json`; full numerical records: `replay_discrepancy_results.json`. The protocol was written before any diagnostic computation. Two DOP853 refinements reduce relative/absolute tolerances and halve the maximum step twice; both independent physical-amount and original uncompiled log-state equations are evaluated.', '',
             'All diagnostic physical, terminal and operational classifications agree with the saved outcomes: **' + str(report['all_replay_classifications_agree_with_saved']) + '**. The original audit remains `hard_checks_passed=false`.', '']
    for episode in report['episodes']:
        arms = episode['arms']
        comparisons = episode['comparisons']
        metric = 'max_physical_amount_absolute_error_g'
        lines.extend(['## ' + episode['case'] + ' / ' + episode['method'], '',
                      '- Original independent audit versus saved endpoints: %.9g g.' % arms['amount_audit']['versus_saved'][metric],
                      '- Finest independent amount versus saved endpoints: %.9g g.' % arms['amount_refine_2']['versus_saved'][metric],
                      '- Finest original uncompiled log versus saved endpoints: %.9g g.' % arms['log_uncompiled_refine_2']['versus_saved'][metric],
                      '- Amount refinement 1 versus 2: %.9g g.' % comparisons['amount_refinement_1_vs_2'][metric],
                      '- Log refinement 1 versus 2: %.9g g.' % comparisons['log_refinement_1_vs_2'][metric],
                      '- Finest amount versus finest log: %.9g g.' % comparisons['finest_amount_vs_finest_log'][metric],
                      '- Repeated original independent replay versus archived replay: %.9g g.' % comparisons['amount_audit_vs_archived_audit'][metric], ''])
    lines.extend(['## Interpretation limits', '',
                  'Convergence across coordinate formulations supports an empirical integration-error diagnosis when refinement and cross-formulation differences are much smaller than the discrepancy from saved primary states. The detailed numbers above must be interpreted before attributing a cause. Neither agreement nor DOP853 error estimates provide a rigorous enclosure. Double precision imposes a roundoff floor; the two requested relative tolerances remain above SciPy\'s approximate 2.22e-14 lower bound. Small errors need not decrease monotonically.', '',
                  'The six episodes were selected because the original audit flagged them; this is not a fresh effectiveness test. Substrate minima are sampled at solver nodes. These offline integrations establish no controller speed claim, robustness guarantee or publication-readiness claim. The original failed audit, thresholds and primary outcomes remain unchanged.', ''])
    NOTE.write_text('\n'.join(lines))
    print(json.dumps({'complete': True, 'episodes': len(report['episodes']),
                      'all_classifications_agree': report['all_replay_classifications_agree_with_saved'],
                      'original_audit_preserved': True, 'elapsed_offline_seconds': report['elapsed_offline_seconds']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--declare', action='store_true')
    group.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    declare() if args.declare else execute()


if __name__ == '__main__':
    main()
