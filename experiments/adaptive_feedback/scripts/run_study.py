"""Closed-loop feed-control experiments with separate development and evaluation paths."""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from model import ROOT, SOURCE, Reactor, initial_state, observable, RATIO_TOL, DT
from controllers import direct, surrogate, estimate_kappa, feasible

METHODS = ['open_loop', 'nominal_mpc', 'adaptive_direct_mpc', 'adaptive_limited_direct',
           'adaptive_ungated', 'adaptive_verified']


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def save(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, default=serial)+'\n')


def observe(true_state, noise):
    y = true_state.copy()
    # Defined observation model, no hidden state supplied to the controller.
    y[1:3] += noise[:2]
    y[3] = max(0.0, y[3]+noise[2]*y[0])
    return y


def episode(case, method, checkpoint=None):
    model = Reactor()
    plant = Reactor()
    truth = initial_state(case['initial_ratio'], case['initial_glucose'])
    observations = np.asarray(case.get('observation_noise', np.zeros((8, 3))))
    state = observe(truth, observations[0])
    kappa = np.ones(2)
    previous_plan = np.full(7, 0.065)
    history = []
    stages = []
    controls = []
    truth_states = [truth.copy()]
    controller_seconds = 0.0
    estimator_seconds = 0.0
    started = time.perf_counter()
    for stage in range(7):
        stage0 = time.perf_counter()
        call0 = model.calls
        start = previous_plan if stage == 0 else previous_plan[1:]
        estimation = {'status': 'nominal_model'}
        try:
            if method.startswith('adaptive'):
                t0 = time.perf_counter()
                try:
                    kappa, estimation = estimate_kappa(model, history, kappa)
                finally:
                    estimator_seconds += time.perf_counter()-t0
            if method == 'open_loop' and stage > 0:
                plan = start
                prediction = None
                info = {'mode': 'fixed_plan'}
                e = None
            elif method in ('open_loop', 'nominal_mpc', 'adaptive_direct_mpc', 'adaptive_limited_direct'):
                plan, prediction, status, e = direct(model, state, kappa, case['target_ratio'], start,
                                                     maxiter=5 if method == 'adaptive_limited_direct' else 45)
                info = {'mode': 'direct', 'status': status}
            else:
                plan, prediction, info, e = surrogate(model, state, kappa, case['target_ratio'], start,
                                                     verified=method == 'adaptive_verified')
        except Exception as ex:
            if checkpoint:
                checkpoint({'case': case, 'method': method, 'completed_stages': stages,
                            'failure_stage': stage, 'error': repr(ex),
                            'controller_seconds': controller_seconds+time.perf_counter()-stage0,
                            'estimator_seconds': estimator_seconds, 'model_calls': model.calls,
                            'model_seconds': model.seconds, 'model_failures': model.failures})
            raise
        duration = time.perf_counter()-stage0
        controller_seconds += duration
        feed = float(plan[0])
        if checkpoint:
            checkpoint({'case': case, 'method': method, 'completed_stages': stages,
                        'pending_stage': stage, 'pending_action': feed,
                        'controller_seconds': controller_seconds, 'estimator_seconds': estimator_seconds,
                        'model_calls': model.calls, 'model_seconds': model.seconds})
        # Hidden plant parameters are accessed only after the action is fixed.
        plant_kappa = case['kappa_after'] if stage >= case.get('change_stage', 99) else case['kappa_before']
        actual = plant.predict(truth, [feed], plant_kappa, tight=True, record=True)
        next_truth = actual['state']
        next_observed = observe(next_truth, observations[stage+1])
        history.append((state.copy(), feed, next_observed.copy()))
        record = {'stage': stage, 'time_h': stage*DT, 'observed_state': state,
                  'kappa_estimate': kappa.copy(), 'estimation': estimation,
                  'plan': plan, 'applied_feed_L_h': feed, 'decision': info,
                  'controller_seconds': duration, 'model_calls': model.calls-call0,
                  'selected_model_feasible': feasible(prediction, case['target_ratio']) if prediction else None,
                  'selected_model_z': prediction['z'] if prediction else None,
                  'selected_model_B': prediction['B'] if prediction else None,
                  'evaluations': [{'control': u, 'prediction': q} for u, q in e.trace] if e else [],
                  'failed_evaluations': e.failures if e else [],
                  'true_start_state': truth, 'true_end_state': next_truth,
                  'true_plant_kappa': plant_kappa, 'plant_min_S': actual['min_S'],
                  'next_substrate_observation_clipped': bool(next_truth[3]/next_truth[0]+observations[stage+1,2] < 0),
                  'plant_max_S': actual['max_S'], 'plant_ramp_rhs_calls': actual['ramp_rhs_calls'],
                  'plant_trajectory': actual['trajectory'], 'plant_times_local': actual['times']}
        stages.append(record)
        if checkpoint:
            checkpoint({'case': case, 'method': method, 'completed_stages': stages,
                        'model_calls': model.calls, 'controller_seconds': controller_seconds})
        controls.append(feed)
        truth_states.append(next_truth.copy())
        previous_plan = plan
        state, truth = next_observed, next_truth
    out = observable(truth)
    logerror = abs(out['log_ratio']-np.log(case['target_ratio']))
    terminal_ok = bool(logerror <= RATIO_TOL and out['volume_L'] <= 1.8+1e-8)
    physical_ok = bool(min(s['plant_min_S'] for s in stages) >= -2e-7
                       and all(-1e-10 <= u <= .2+1e-10 for u in controls)
                       and all(y[0] <= 1.8+1e-8 and np.all(np.isfinite(y)) for y in truth_states))
    out.update(log_ratio_error=logerror,
               ratio_relative_error_percent=100*abs(out['ratio']/case['target_ratio']-1),
               terminal_ratio_volume_feasible=terminal_ok,
               physical_path_feed_valid=physical_ok,
               terminal_feasible=terminal_ok and physical_ok,
               terminal_ratio_violation=max(0.0, logerror-RATIO_TOL),
               minimum_substrate=min(s['plant_min_S'] for s in stages),
               maximum_substrate=max(s['plant_max_S'] for s in stages))
    return {'case': case, 'method': method, 'result': out, 'controls': controls,
            'true_states': truth_states, 'stages': stages,
            'controller_seconds': controller_seconds, 'estimator_seconds': estimator_seconds,
            'model_calls': model.calls, 'model_seconds': model.seconds,
            'model_failures': model.failures, 'plant_verification_seconds': plant.seconds,
            'wall_seconds': time.perf_counter()-started}


def pilot_cases():
    specs = [('nominal', [1., 1.], [1., 1.], 99),
             ('constant_mismatch', [.85, 1.15], [.85, 1.15], 99),
             ('late_shift', [1., 1.], [1.15, .85], 3)]
    return [dict(id='development_'+name, initial_ratio=1.0, initial_glucose=20.0,
                 target_ratio=1.0, kappa_before=before, kappa_after=after, change_stage=change)
            for name, before, after, change in specs]


def hashes():
    paths = [ROOT/'scripts'/name for name in ('model.py', 'controllers.py', 'run_study.py')]+[SOURCE]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pilot', 'evaluation'], default='pilot')
    ap.add_argument('--label', default='pilot_v1')
    ap.add_argument('--methods', nargs='+', default=METHODS)
    ap.add_argument('--case-limit', type=int)
    args = ap.parse_args()
    destination = ROOT/'results'/args.label
    destination.mkdir(exist_ok=True)
    if args.mode == 'pilot':
        cases = pilot_cases()
    else:
        protocol = json.loads((ROOT/'protocol.json').read_text())
        lock = json.loads((ROOT/'LOCK.json').read_text())
        if args.case_limit is not None or args.methods != protocol['methods']:
            raise RuntimeError('Locked evaluation requires all prespecified cases and methods')
        for p, expected in lock['hashes'].items():
            path = Path(p) if Path(p).is_absolute() else ROOT/p
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Locked source changed: '+p)
        cases = protocol['cases']
    if args.case_limit:
        cases = cases[:args.case_limit]
    execution = {'mode': args.mode, 'hashes': hashes(),
          'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__,
          'scipy': scipy.__version__, 'platform': platform.platform(), 'cases': cases,
          'methods': args.methods, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if any(m not in METHODS for m in args.methods):
        raise ValueError('Unknown controller method')
    manifest_path = destination/'execution.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ('mode', 'hashes', 'cases', 'methods'):
            if old[key] != execution[key]:
                raise RuntimeError('Existing run has different '+key+'; use a new label')
    else:
        save(manifest_path, execution)
        snapshot = destination/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in ('model.py', 'controllers.py', 'run_study.py'):
            shutil.copy2(ROOT/'scripts'/name, snapshot/name)
        if args.mode == 'evaluation':
            shutil.copy2(ROOT/'protocol.json', snapshot/'protocol.json')
            shutil.copy2(ROOT/'LOCK.json', snapshot/'LOCK.json')
    rows = []
    for i, case in enumerate(cases):
        # Rotate execution order to reduce systematic timing/order association.
        methods = args.methods[i % len(args.methods):]+args.methods[:i % len(args.methods)]
        for method in methods:
            path = destination/f'{case["id"]}__{method}.json'
            identity = {'source_hashes': execution['hashes'], 'mode': args.mode,
                        'case_sha256': hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()}
            if path.exists():
                row = json.loads(path.read_text())
                if row.get('provenance') != identity or row['method'] != method or row['case'] != case:
                    raise RuntimeError('Existing episode identity mismatch: '+str(path))
            else:
                partial = destination/f'{case["id"]}__{method}.partial.json'
                attempt0 = time.perf_counter()
                try:
                    row = episode(case, method, checkpoint=lambda content: save(partial, content))
                    row['status'] = 'completed'
                except Exception as ex:
                    prefix = json.loads(partial.read_text()) if partial.exists() else {}
                    row = {'case': case, 'method': method, 'status': 'failed', 'error': repr(ex),
                           'partial_record': prefix, 'controller_seconds': prefix.get('controller_seconds', 0.0),
                           'attempt_seconds': time.perf_counter()-attempt0,
                           'result': {'terminal_feasible': False, 'ratio_relative_error_percent': None,
                                      'biomass_amount_gDW': None}}
                row['provenance'] = identity
                save(path, row)
                if row['status'] == 'completed' and partial.exists(): partial.unlink()
            rows.append(row)
            print(case['id'], method, row['result']['terminal_feasible'],
                  row['result']['ratio_relative_error_percent'],
                  row['result']['biomass_amount_gDW'],
                  round(row['controller_seconds'], 3), flush=True)
    save(destination/'complete.json', {'episodes': len(rows), 'expected': len(cases)*len(args.methods),
                                      'status': 'complete', 'hashes': hashes()})


if __name__ == '__main__': main()
