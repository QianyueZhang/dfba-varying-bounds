"""Offline local-optimizer budget diagnostic; never changes a main decision.

Preparation only by default. Execution is deliberately gated on completed
effectiveness and serial timing panels. Frozen core modules are imported only
after these guards pass. The three arms run sequentially, with separate cost.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'audit' / 'optimization_budget_protocol.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, default=serial) + '\n')
    temp.replace(path)


def metrics(theta, evaluation):
    log_biomass = np.asarray(evaluation['logB'], dtype=float)
    return {
        'theta': np.asarray(theta), 'gain': float(theta[-1]),
        'finite_model_feasible': bool(evaluation['feasible']),
        'violation': float(evaluation['violation']),
        'merit': float(evaluation['merit']),
        'mean_log_biomass': float(np.mean(log_biomass)),
        'mean_biomass_gDW': float(np.mean(np.exp(log_biomass))),
        'scenario_z': np.asarray(evaluation['z']),
        'scenario_log_biomass': log_biomass,
    }


def new_evaluation(inputs):
    model = Reactor(inputs['family'])
    scenarios = [(item['name'], np.asarray(item['kappas'], dtype=float))
                 for item in inputs['scenarios']]
    evaluator = Evaluation(model, np.asarray(inputs['observed_state'], dtype=float),
                           inputs['target_ratio'], scenarios,
                           np.asarray(inputs['reference'], dtype=float),
                           np.asarray(inputs['basis'], dtype=float))
    return model, evaluator


def run_arm(inputs, name, start, gain, initial_candidates):
    """Include construction/prefill in offline elapsed time; keep every trace."""
    started = time.perf_counter()
    model, evaluator = new_evaluation(inputs)
    record = {'id': name, 'maxiter': 60, 'gain_enabled': bool(gain),
              'start': np.asarray(start), 'initial_candidates': [], 'status': 'running'}
    theta = None
    try:
        for label, candidate in initial_candidates:
            result = evaluator(np.asarray(candidate, dtype=float))
            record['initial_candidates'].append({'label': label, **metrics(candidate, result)})
        theta, result, optimizer_status = direct(evaluator, np.asarray(start, dtype=float), gain, 60)
        record.update(status='completed', optimizer_status=optimizer_status,
                      selected=metrics(theta, result), selected_evaluation=result)
    except Exception as error:
        record.update(status='failed', error=repr(error))
        if evaluator.trace:
            theta, result = evaluator.best()
            record['retained_after_failure'] = metrics(theta, result)
        # A failed arm does not silently seed a dependent optimization.
        theta = None
    finally:
        record.update(offline_seconds=time.perf_counter() - started,
                      model_seconds=model.seconds, model_calls=model.calls,
                      model_failures=model.failures,
                      evaluations=[{'theta': value, 'evaluation': result}
                                   for value, result in evaluator.trace],
                      evaluation_failures=evaluator.failures)
    return record, theta


def identity_guard(protocol):
    if digest(ROOT / 'protocol.json') != protocol['primary_protocol_sha256']:
        raise RuntimeError('Primary protocol identity differs from diagnostic specification')
    lock = json.loads((ROOT / 'LOCK.json').read_text())
    for relative, expected in lock['hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError('Locked source changed: ' + relative)
    completed = {}
    for campaign in protocol['required_completed_campaigns']:
        folder = ROOT / 'results' / campaign
        path = folder / 'complete.json'
        if not path.exists():
            raise RuntimeError('Do not run during campaign/timing: missing ' + str(path))
        if list(folder.glob('*.partial.json')):
            raise RuntimeError('Unfinished episode checkpoint remains in ' + str(folder))
        record = json.loads(path.read_text())
        if record.get('episodes') != record.get('expected'):
            raise RuntimeError('Incomplete campaign inventory: ' + campaign)
        completed[campaign] = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                               'episodes': record['episodes']}
    return lock, completed


def get_inputs(case, ordinal, protocol):
    path = ROOT / 'results' / protocol['source_campaign'] / (
        case['id'] + '__' + protocol['source_method'] + '.json')
    episode = json.loads(path.read_text())
    if episode.get('case') != case or episode.get('method') != protocol['source_method']:
        raise RuntimeError('Source episode case/method mismatch')
    execution = json.loads((path.parent / 'execution.json').read_text())
    if episode.get('source_hashes') != execution.get('hashes'):
        raise RuntimeError('Source episode hashes differ from its campaign')
    if episode.get('status') != 'completed':
        raise RuntimeError('Specified source episode did not complete')
    stage = episode['stages'][protocol['source_stage']]
    if stage['stage'] != 0:
        raise RuntimeError('Diagnostic must use stage zero')
    decision = stage['decision']
    if 'controller_exception' in decision or not decision.get('trace'):
        raise RuntimeError('No valid first-decision trace for specified diagnostic')
    initial_theta = np.asarray(decision['trace'][0]['theta'], dtype=float)
    if initial_theta[-1] != 0.:
        raise RuntimeError('First trace is not a zero-gain incumbent')
    # Deliberately pass only controller-available fields to the optimizer.
    inputs = {'family': case['family'], 'observed_state': stage['observed_state'],
              'target_ratio': case['target_ratio'], 'reference': decision['reference'],
              'basis': decision['basis'], 'scenarios': decision['scenarios']}
    return {'case_id': case['id'], 'case_ordinal': ordinal, 'stage': 0,
            'source_episode': str(path.relative_to(ROOT)), 'source_episode_sha256': digest(path),
            'inputs': inputs, 'initial_theta': initial_theta,
            'saved_12': metrics(decision['theta'], decision['selected']),
            'saved_12_optimizer_status': decision['status'],
            'saved_12_controller_seconds_parallel_diagnostic_only': stage['controller_seconds'],
            'saved_12_evaluations': decision['trace']}


def evaluate_case(source, output):
    inputs = source['inputs']
    original = source['initial_theta']
    result = dict(source)
    preparation = time.perf_counter()
    model, evaluator = new_evaluation(inputs)
    replay = evaluator(np.asarray(source['saved_12']['theta'], dtype=float))
    result['saved_12_replay'] = metrics(source['saved_12']['theta'], replay)
    result['saved_12_replay_evaluation'] = replay
    result['saved_12_replay_max_abs_z_difference'] = float(np.max(np.abs(
        np.asarray(replay['z']) - source['saved_12']['scenario_z'])))
    result['saved_12_replay_max_abs_logB_difference'] = float(np.max(np.abs(
        np.asarray(replay['logB']) - source['saved_12']['scenario_log_biomass'])))
    result['preparation_and_replay_seconds'] = time.perf_counter() - preparation
    result['preparation_model_calls'] = model.calls
    result['arms'] = []
    save(output, result)

    zero_record, zero_theta = run_arm(inputs, 'gain_zero_60', original, False,
                                      [('original_zero_gain', original)])
    result['arms'].append(zero_record)
    save(output, result)
    original_record, _ = run_arm(inputs, 'gain_original_60', original, True,
                                 [('original_zero_gain', original)])
    result['arms'].append(original_record)
    save(output, result)
    if zero_theta is None:
        seeded_record = {'id': 'gain_seeded_60', 'status': 'skipped_dependency_failure',
                         'reason': 'No completed gain_zero_60 candidate available',
                         'offline_seconds': 0., 'model_calls': 0, 'model_failures': 0}
    else:
        seeded_record, _ = run_arm(inputs, 'gain_seeded_60', zero_theta, True,
                                   [('original_zero_gain', original),
                                    ('gain_zero_60_selected', zero_theta)])
    result['arms'].append(seeded_record)
    result['offline_seconds'] = result['preparation_and_replay_seconds'] + sum(
        arm['offline_seconds'] for arm in result['arms'])
    result['complete'] = True
    save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='Run only after main campaign and serial timing are complete')
    parser.add_argument('--label', default='optimization_budget_results')
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if not args.execute:
        print(json.dumps({'prepared_only': True, 'protocol': str(PROTOCOL_PATH),
                          'case_ordinals': protocol['zero_based_case_ordinals'],
                          'execution': protocol['execution']}, indent=2))
        return
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.label):
        raise ValueError('Output label must contain only letters, digits, underscores or hyphens')
    lock, completed = identity_guard(protocol)
    destination = ROOT / 'audit' / args.label
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('Refusing to overwrite prior diagnostic; use a fresh label')

    global np, Reactor, Evaluation, direct
    import numpy as np
    sys.path.insert(0, str(ROOT / 'scripts'))
    from model import Reactor
    from controllers import Evaluation, direct
    import kernel

    main_protocol = json.loads((ROOT / 'protocol.json').read_text())
    ordinals = protocol['zero_based_case_ordinals']
    cases = [main_protocol['cases'][i] for i in ordinals]
    if [case['id'] for case in cases] != protocol['case_ids']:
        raise RuntimeError('Selected case ordinal/identity mismatch')
    sources = [get_inputs(case, i, protocol) for i, case in zip(ordinals, cases)]
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': protocol, 'protocol_sha256': digest(PROTOCOL_PATH),
                'script_sha256': digest(__file__), 'primary_lock': lock,
                'completed_campaigns': completed, 'native_kernel_available': kernel.AVAILABLE,
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'scope': 'Post hoc, offline, no main-action modification, no global optimality claim'}
    save(destination / 'execution.json', manifest)
    (destination / 'source_script.py').write_text(Path(__file__).read_text())
    records = []
    started = time.perf_counter()
    for source in sources:
        path = destination / (source['case_id'] + '.json')
        try:
            result = evaluate_case(source, path)
            records.append({'case_id': source['case_id'], 'file': path.name,
                            'status': 'completed', 'saved_12': result['saved_12'],
                            'arms': [{key: arm[key] for key in
                                      ('id', 'status', 'offline_seconds', 'model_calls',
                                       'model_failures', 'selected', 'optimizer_status') if key in arm}
                                     for arm in result['arms']],
                            'offline_seconds': result['offline_seconds']})
        except Exception as error:
            failure = {'case_id': source['case_id'], 'status': 'failed', 'error': repr(error)}
            records.append(failure)
            save(destination / (source['case_id'] + '.error.json'), failure)
        save(destination / 'summary.json', {'complete': False, 'cases': records})
        print(json.dumps({'case': source['case_id'], 'status': records[-1]['status']}), flush=True)
    save(destination / 'summary.json', {'complete': True, 'cases': records,
         'total_offline_wall_seconds': time.perf_counter() - started,
         'interpretation': protocol['interpretation']})
    print(json.dumps({'complete': True, 'output': str(destination)}))


if __name__ == '__main__':
    main()
