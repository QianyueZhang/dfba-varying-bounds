"""Lightweight failure-routing tests with deliberately non-biological stubs.

No ODE integration or optimizer is run. These tests verify control flow,
recording, and endpoint labels only; they are not scientific outcome cases.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import controllers
import run_study


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class CandidateFailureModel:
    """Return transparent algebraic records, failing only the test proposal."""
    def __init__(self):
        self.calls = 0
        self.failures = 0

    def policy(self, state, theta, kappas, reference, basis):
        self.calls += 1
        theta = np.asarray(theta)
        if theta[0] > .055:
            self.failures += 1
            raise RuntimeError('injected_candidate_verification_failure')
        n = len(kappas)
        return {
            'state': state.copy(), 'z': 0., 'logB': 0., 'B': 1.,
            'V': float(state[0]), 'min_S': float(state[3] / state[0]),
            'feeds': theta[:n].copy(),
            'knots': np.tile(state, (n + 1, 1)),
            'jz': np.r_[np.ones(n), 0.],
            'jlogB': np.zeros(n + 1),
        }


def check_managed_candidate_failure():
    model = CandidateFailureModel()
    state = run_study.initial_state(1., 20.)
    calls = []

    def injected_proposal(evaluator, center, radius, gain):
        candidate = center.copy()
        candidate[0] = .06
        return candidate, {'success': True, 'radius': radius}

    def fallback_spy(evaluator, center, gain, maxiter):
        calls.append({'gain': gain, 'maxiter': maxiter,
                      'center': center.copy(), 'failures_seen': len(evaluator.failures)})
        theta, result = evaluator.best()
        return theta, result, {'success': True, 'message': 'stub_direct_fallback'}

    with patch.object(controllers, 'proposal', injected_proposal), \
         patch.object(controllers, 'direct', fallback_spy):
        plan, info = controllers.solve(model, state, np.ones(2), 1.,
                                       np.full(2, .05), 5, 'scenario_managed')

    assert info['fallback'] is True
    assert len(calls) == 1 and calls[0]['maxiter'] == 12
    assert calls[0]['gain'] is True and calls[0]['failures_seen'] == 1
    assert len(info['events']) == 1
    assert info['events'][0]['action'] == 'verification_error'
    assert 'injected_candidate_verification_failure' in info['events'][0]['error']
    assert len(info['failures']) == 1
    assert info['failures'][0]['theta'][0] == .06
    assert np.array_equal(plan, np.full(2, .05))
    assert model.failures == 1 and model.calls == 5
    return {'passed': True, 'fallback_calls': calls,
            'event': info['events'][0], 'failure_record': info['failures'][0],
            'returned_plan': plan, 'model_calls': model.calls,
            'model_failures': model.failures,
            'scope': 'Candidate evaluation failure routes to retained-incumbent direct fallback; direct solver is a spy.'}


class StaticPhysicalPlant:
    """Inert material-balance stub; no growth or substrate consumption."""
    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.failures = 0
        self.seconds = 0.

    def predict(self, state, controls, kappa, *, tight, record):
        self.calls += 1
        assert tight is True and record is True
        assert len(controls) == 1 and controls[0] == 0.
        result = np.array(state, dtype=float).copy()
        result[0] += run_study.DT * controls[0]
        result[3] += 100. * run_study.DT * controls[0]
        return {'state': result, 'min_S': float(result[3] / result[0]),
                'max_S': float(result[3] / result[0])}


def check_whole_solver_failure():
    attempts = []
    checkpoints = []

    def injected_solve(model, state, kappa, target, start, stage, method):
        attempts.append({'stage': stage, 'horizon': len(start), 'method': method})
        raise RuntimeError('injected_whole_controller_failure')

    def prior_estimator(model, history, previous):
        return np.asarray(previous).copy(), {'status': 'stub_prior'}

    case = {'id': 'routing_only_inert_plant', 'family': 'table',
            'initial_ratio': 1., 'initial_glucose': 20., 'target_ratio': 1.,
            'kappa_before': [1., 1.], 'kappa_after': [1., 1.], 'change_stage': 99}
    with patch.object(run_study, 'Reactor', StaticPhysicalPlant), \
         patch.object(run_study, 'solve', injected_solve), \
         patch.object(run_study, 'estimate_kappa', prior_estimator):
        row = run_study.episode(case, 'scenario_managed', checkpoints.append)

    assert [item['stage'] for item in attempts] == list(range(7))
    assert [item['horizon'] for item in attempts] == list(range(7, 0, -1))
    assert len(row['stages']) == 7 and len(checkpoints) == 14
    assert all(item['feed'] == 0. for item in row['stages'])
    assert all('injected_whole_controller_failure' in item['decision']['controller_exception']
               for item in row['stages'])
    assert all(item['true_end_state'][0] == .5 for item in row['stages'])
    assert row['result']['physical_valid'] is True
    assert row['result']['terminal_feasible'] is True
    assert row['result']['controller_failure'] is True
    assert row['result']['operational_success'] is False
    assert row['controller_seconds'] >= row['estimator_seconds'] >= 0.
    assert row['controller_seconds'] == sum(item['controller_seconds'] for item in row['stages'])
    assert checkpoints[-1]['controller_seconds'] == row['controller_seconds']
    return {'passed': True, 'solver_attempts': attempts, 'checkpoint_count': len(checkpoints),
            'executed_feeds': [item['feed'] for item in row['stages']],
            'result': row['result'], 'controller_seconds': row['controller_seconds'],
            'estimator_seconds': row['estimator_seconds'],
            'scope': 'Whole-solver exceptions execute zero-feed backup, preserve logs/timing, and separate physical endpoint from operational success.'}


def main():
    report = {'purpose': 'Lightweight monkeypatched failure-routing checks; not biological or numerical validation',
              'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'source_hashes': {}, 'tests': {}, 'passed': False}
    for rel in ['scripts/controllers.py', 'scripts/run_study.py', 'scripts/model.py',
                'scripts/base_model.py', 'scripts/kernel.py', 'audit/test_failure_routing.py']:
        path = ROOT / rel
        if path.exists():
            report['source_hashes'][rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        report['tests']['managed_candidate_verification_error'] = check_managed_candidate_failure()
        report['tests']['whole_controller_error_backup'] = check_whole_solver_failure()
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        destination = ROOT / 'audit' / 'failure_routing_results.json'
        destination.write_text(json.dumps(report, indent=2, default=jsonable) + '\n')
        print(json.dumps({'passed': report['passed'], 'tests': len(report['tests']),
                          'output': str(destination)}))


if __name__ == '__main__':
    main()
