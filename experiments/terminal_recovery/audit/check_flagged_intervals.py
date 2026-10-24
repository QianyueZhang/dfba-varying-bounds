"""Post hoc refinement of every non-prefix failed matched interval.

The original deployment audit remains unchanged and failed. These are isolated
interval diagnostics, not a rerun of a controller or a revised success score.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'audit'
OUTPUT = HERE / 'check_flagged_intervals.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value):
    temporary = Path(str(OUTPUT)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(OUTPUT)


def declare():
    if OUTPUT.exists():
        raise RuntimeError('Diagnostic already declared; do not overwrite')
    source = json.loads((HERE/'deployment_results.json').read_text())
    if not source['complete']:
        raise RuntimeError('Complete original deployment first')
    selected = sorted({failure['transition_id'] for failure in source['failures']
        if failure.get('scope') == 'matched_interval' and 'transition_id' in failure
        and not source['transitions'][failure['transition_id']]['logical_uses'].get('prefix', 0)})
    if selected != [960, 1291]:
        raise RuntimeError('Observed failed interval set differs from assigned two-record diagnostic')
    paths = ['audit/deployment_results.json', 'audit/deployment_protocol.json', 'audit/check_flagged_intervals.py',
             'scripts/engine.py', 'vendor/prior_kernel.py', 'audit/independent_amount_model.py', 'data/flux_table_eps1e-06.json']
    protocol = dict(status='Post hoc, declared before diagnostic execution',
        declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), selected_transition_ids=selected,
        selection='Every failed matched interval not used by a prefix; outcome-selected numerical diagnosis only',
        inputs_sha256={p: sha(ROOT/p) for p in paths},
        arms={
            'amount_original': dict(kind='amount', rtol=2e-13, atol=2e-14, maxstep=.025),
            'amount_refined': dict(kind='amount', rtol=2e-14, atol=2e-15, maxstep=.005),
            'engine_original': dict(kind='engine', rtol=2e-11, atol=2e-13, maxstep=.05),
            'engine_refined': dict(kind='engine', rtol=2e-13, atol=2e-15, maxstep=.01)},
        scipy_rtol_note='Requested refined rtol=2e-14 is below SciPy floor; SciPy clips it to approximately2.220446049250313e-14. Record warnings and effective value.',
        thresholds=source['protocol']['thresholds'],
        comparisons=['original amount versus archived amount', 'original engine versus archived engine',
                     'original amount versus refined amount', 'original engine versus refined engine',
                     'refined engine versus refined amount', 'original engine versus refined amount'],
        rule='Use the same archived independent initial amount state, feed, true affinity and inhibition for every arm. Never replan, substitute roots, alter original results or relax thresholds.',
        limits='Isolated one-interval comparisons only; do not infer unchanged refined full-policy classifications, global error enclosures or a retrospective audit pass.')
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    save(dict(status='declared', complete=False, protocol=protocol, protocol_sha256=protocol_hash, intervals=[]))
    print(json.dumps(dict(declared=True, selected=selected, numerical_execution=False)))


def execute():
    report = json.loads(OUTPUT.read_text())
    if report['status'] != 'declared':
        raise RuntimeError('Diagnostic is not in the unexecuted declared state')
    protocol = report['protocol']
    for p, expected in protocol['inputs_sha256'].items():
        if sha(ROOT/p) != expected:
            raise RuntimeError('Declared input changed: '+p)
    sys.path.insert(0, str(ROOT/'scripts'))
    sys.path.insert(0, str(HERE))
    import numpy as np
    import scipy
    import engine
    import independent_amount_model as independent
    source = json.loads((HERE/'deployment_results.json').read_text())
    report.update(status='running', scipy_version=scipy.__version__,
                  scipy_effective_refined_rtol=float(max(2e-14, 100*np.finfo(float).eps)))
    save(report)
    started = time.perf_counter()

    def compare(a, b):
        a, b = np.asarray(a), np.asarray(b)
        err = np.abs(a-b)
        thresholds = protocol['thresholds']
        allowed = thresholds['amount_absolute_g']+thresholds['amount_relative']*np.abs(b[1:])
        logerr = float(abs(np.log(a[1]/a[2])-np.log(b[1]/b[2])))
        return dict(component_absolute_errors=err.tolist(), maximum_amount_absolute_error=float(err[1:].max()),
                    maximum_amount_error_over_original_allowance=float(np.max(err[1:]/allowed)),
                    log_ratio_error=logerr,
                    within_original_comparison_thresholds=bool(np.all(err[1:]<=allowed) and err[0]<=thresholds['volume_absolute_L'] and logerr<=thresholds['log_ratio_absolute']))

    for ident in protocol['selected_transition_ids']:
        record = source['transitions'][ident]
        initial = np.asarray(record['input_amount_state'])
        item = dict(transition_id=ident, original_record=record, arms={})
        for name, setting in protocol['arms'].items():
            clock = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if setting['kind'] == 'amount':
                        independent.RTOL, independent.ATOL, independent.MAX_STEP = setting['rtol'], setting['atol'], setting['maxstep']
                        model = independent.AmountReactor(record['family'], record['inhibition'])
                        out = model.interval(initial, record['feed'], record['kappa'])
                        endpoint, minimum = out['amount_state'], out['min_S']
                        work = dict(rhs_calls=out['rhs_calls'])
                    else:
                        endpoint, minimum, accepted, rejected = engine.flow(independent.to_log(initial), record['feed'],
                            np.asarray(record['kappa']), engine.arguments(record['family'], record['inhibition']),
                            setting['rtol'], setting['atol'], setting['maxstep'])
                        endpoint = independent.from_log(endpoint)
                        work = dict(accepted_steps=int(accepted), rejected_steps=int(rejected))
                item['arms'][name] = dict(complete=True, amount_state=endpoint.tolist(), min_S=float(minimum),
                    seconds=time.perf_counter()-clock, warnings=[str(w.message) for w in caught], **work)
            except Exception as error:
                item['arms'][name] = dict(complete=False, exception_type=type(error).__name__, exception=str(error), seconds=time.perf_counter()-clock)
        if all(a['complete'] for a in item['arms'].values()):
            a = {k:v['amount_state'] for k,v in item['arms'].items()}
            item['comparisons'] = dict(
                original_amount_vs_archived=compare(a['amount_original'], record['independent']['amount_state']),
                original_engine_vs_archived=compare(a['engine_original'], record['engine']['amount_state']),
                amount_original_vs_refined=compare(a['amount_original'], a['amount_refined']),
                engine_original_vs_refined=compare(a['engine_original'], a['engine_refined']),
                refined_engine_vs_refined_amount=compare(a['engine_refined'], a['amount_refined']),
                original_engine_vs_refined_amount=compare(a['engine_original'], a['amount_refined']))
        report['intervals'].append(item)
        save(report)
        print('diagnosed transition', ident, flush=True)
    report['input_hashes_preserved'] = all(sha(ROOT/p)==h for p,h in protocol['inputs_sha256'].items())
    report['original_deployment_passed'] = source['passed']
    report['elapsed_seconds'] = time.perf_counter()-started
    report['complete'] = all(a['complete'] for item in report['intervals'] for a in item['arms'].values()) and len(report['intervals'])==2
    report['status'] = 'complete' if report['complete'] else 'incomplete'
    save(report)
    print(json.dumps(dict(complete=report['complete'], original_deployment_passed=source['passed'], elapsed_seconds=report['elapsed_seconds'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--declare', action='store_true')
    modes.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    declare() if args.declare else execute()


if __name__ == '__main__':
    main()
