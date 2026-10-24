"""Direct and local-affine predictive controllers with parameter estimation.

Trust-region proposals, nonlinear candidate checks and fallback optimisation
are established methods. Their performance is assessed for the stated reactor."""
import time
import numpy as np
from scipy.optimize import minimize, linprog, least_squares
from model import DT, VOLUME_CAP, RATIO_TOL, DESIGN_TOL


def feasible(q, target, tolerance=DESIGN_TOL):
    return (abs(q['z']-np.log(target)) <= tolerance+1e-7
            and q['V'] <= VOLUME_CAP+1e-8 and q['min_S'] >= -2e-7)


def violation(q, target):
    return max(0.0, abs(q['z']-np.log(target))-DESIGN_TOL) + max(0.0, q['V']-VOLUME_CAP)


def scaled_start(start, state):
    u = np.clip(np.asarray(start, float), 0.0, 0.2)
    capacity = max(0.0, (VOLUME_CAP-state[0])/DT)
    if u.sum() > capacity:
        u *= capacity/max(u.sum(), 1e-15)
    return u


class Evaluation:
    def __init__(self, reactor, state, kappa, target):
        self.reactor, self.state, self.kappa, self.target = reactor, state, kappa, target
        self.cache = {}
        self.trace = []
        self.failures = []

    def __call__(self, u):
        u = np.asarray(u, float)
        key = u.tobytes()
        if key not in self.cache:
            started = time.perf_counter()
            try:
                q = self.reactor.predict(self.state, u, self.kappa, gradients=True)
            except Exception as ex:
                self.failures.append({'control': u.copy(), 'error': repr(ex),
                                      'seconds': time.perf_counter()-started})
                raise
            q['evaluation_seconds'] = time.perf_counter()-started
            self.cache[key] = q
            self.trace.append((u.copy(), q))
        return self.cache[key]

    def best(self):
        good = [(u, q) for u, q in self.trace if feasible(q, self.target)]
        if good:
            return max(good, key=lambda x: x[1]['logB'])
        return min(self.trace, key=lambda x: (violation(x[1], self.target), -x[1]['logB']))


def direct(reactor, state, kappa, target, start, maxiter=45, evaluation=None):
    e = evaluation or Evaluation(reactor, state, kappa, target)
    u0 = scaled_start(start, state)
    zt = np.log(target)
    cons = [
        {'type': 'ineq', 'fun': lambda u: VOLUME_CAP-state[0]-DT*np.sum(u),
         'jac': lambda u: -DT*np.ones(len(u))},
        {'type': 'ineq', 'fun': lambda u: DESIGN_TOL-(e(u)['z']-zt),
         'jac': lambda u: -e(u)['jz']},
        {'type': 'ineq', 'fun': lambda u: DESIGN_TOL+(e(u)['z']-zt),
         'jac': lambda u: e(u)['jz']},
    ]
    status = {}
    try:
        e(u0)
        answer = minimize(lambda u: -e(u)['logB'], u0,
                          jac=lambda u: -e(u)['jlogB'], method='SLSQP',
                          bounds=[(0.0, 0.2)]*len(u0), constraints=cons,
                          options={'maxiter': maxiter, 'ftol': 1e-6})
        status = {'success': bool(answer.success), 'message': str(answer.message),
                  'iterations': int(answer.nit)}
    except Exception as ex:
        status = {'success': False, 'message': repr(ex)}
    if not e.trace:
        raise RuntimeError('No valid controller prediction: '+str(status))
    u, q = e.best()
    return u, q, status, e


def estimate_kappa(reactor, history, previous):
    """Fit only observed past intervals. No hidden plant parameter is accepted."""
    if not history:
        return np.asarray(previous), {'status': 'prior', 'n_intervals': 0}
    recent = history[-2:]
    prior = np.asarray(previous, float)
    def residual(k):
        out = []
        for before, feed, after in recent:
            prediction = reactor.predict(before, [feed], k)['state']
            out.extend((prediction[1:3]-after[1:3])/0.01)
            out.append((prediction[3]/prediction[0]-after[3]/after[0])/0.1)
        out.extend(0.05*(k-prior))
        return np.asarray(out)
    answer = least_squares(residual, prior, bounds=([0.75, 0.75], [1.25, 1.25]),
                           max_nfev=18, ftol=1e-7, xtol=1e-7, gtol=1e-7,
                           diff_step=2e-4)
    sv = np.linalg.svd(answer.jac[:-2], compute_uv=False)
    return answer.x, {'status': str(answer.message), 'success': bool(answer.success),
                      'n_intervals': len(recent), 'cost': float(answer.cost),
                      'function_evaluations': int(answer.nfev),
                      'data_jacobian_singular_values': sv.tolist(),
                      'at_bound': bool(np.any(answer.x < 0.7501) or np.any(answer.x > 1.2499))}


def local_proposal(e, center, state, target, radius):
    q = e(center)
    n = len(center)
    # Slack is a declared feasibility-restoration variable, not hidden clipping.
    upper = np.array([min(0.2, x+radius) for x in center])
    lower = np.maximum(0.0, center-radius)
    A = [np.r_[q['jz'], -1.0], np.r_[-q['jz'], -1.0], np.r_[DT*np.ones(n), 0.0]]
    b = [np.log(target)+DESIGN_TOL-q['z']+q['jz']@center,
         -np.log(target)+DESIGN_TOL+q['z']-q['jz']@center,
         VOLUME_CAP-state[0]]
    lp = linprog(np.r_[-q['jlogB'], 100.0], A_ub=A, b_ub=b,
                 bounds=list(zip(lower, upper))+[(0.0, None)], method='highs')
    if not lp.success:
        return center.copy(), {'lp_success': False, 'message': str(lp.message)}
    proposed = lp.x[:n]
    d = proposed-center
    return proposed, {'lp_success': True, 'slack': float(lp.x[-1]),
                       'predicted_z': float(q['z']+q['jz']@d),
                       'predicted_logB': float(q['logB']+q['jlogB']@d),
                       'radius': radius}


def surrogate(reactor, state, kappa, target, start, verified=True, max_corrections=2):
    center = scaled_start(start, state)
    e = Evaluation(reactor, state, kappa, target)
    original = e(center)
    radius = 0.035
    events = []
    fallback = False
    # Ungated control uses the same affine model and LP but skips nonlinear gate.
    if not verified:
        proposal, info = local_proposal(e, center, state, target, radius)
        return proposal, None, {'events': [dict(info, action='unchecked_proposal',
                                               center=center.copy(), proposal=proposal.copy())],
                                    'fallback': False, 'mode': 'unchecked'}, e
    accepted = False
    for attempt in range(max_corrections+1):
        try:
            proposal, info = local_proposal(e, center, state, target, radius)
            q = e(proposal)
        except Exception as ex:
            events.append({'action': 'verification_error', 'attempt': attempt,
                           'error': repr(ex), 'center': center.copy()})
            break
        base = e(center)
        error = abs(q['z']-info.get('predicted_z', base['z']))
        # Model-relative gate: true nonlinear controller prediction must be
        # feasible and cannot lose more than 0.2% biomass vs a feasible incumbent.
        gate = feasible(q, target) and (not feasible(base, target)
                                        or q['logB'] >= base['logB']-0.002)
        events.append(dict(info, action='accept' if gate else 'reject',
                           center=center.copy(), proposal=proposal.copy(),
                           attempt=attempt, actual_z=q['z'], actual_logB=q['logB'],
                           terminal_surrogate_error=error,
                           nonlinear_violation=violation(q, target)))
        if gate:
            center = proposal
            accepted = True
            break
        # Re-linearize at the lowest-violation observed point and contract.
        center, _ = e.best()
        radius *= 0.5
    if not accepted:
        fallback = True
        center, q, status, e = direct(reactor, state, kappa, target, center, evaluation=e)
        events.append({'action': 'direct_fallback', 'status': status})
    else:
        q = e(center)
    return center, q, {'events': events, 'fallback': fallback,
                       'mode': 'verified', 'model_feasible': feasible(q, target)}, e
