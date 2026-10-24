"""Analytic two-flux test of the moving-bound finite-barrier derivative.

The saved comparison uses a closed-form reference, an independent scalar root
solve, and the full equality-constrained differentiated KKT system. Deliberately
missing derivative terms are fault diagnostics, not candidate methods.
"""
from pathlib import Path
from hashlib import sha256
import json
import platform
import sys
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
A = np.array([[1., -1.]])
C = np.array([.7, .3])
B = np.zeros(1)
HORIZON = 12.
MU = .2
TIMES = np.linspace(0., HORIZON, 601)


def bounds(t):
    t = np.asarray(t)
    return .35 + .25 * np.sin(.8 * t), 1.65 + .30 * np.cos(.6 * t + .2)


def bound_rates(t):
    return .2 * np.cos(.8 * t), -.18 * np.sin(.6 * t + .2)


def analytic_value(lower, upper, mu):
    """Stable closed-form root after eliminating v1=v2=q.

    Effective barrier coefficient k=2 mu, linear objective coefficient Csum=1.
    Compute the upper slack by rationalising its quadratic expression.
    """
    width = np.asarray(upper) - np.asarray(lower)
    k = 2. * mu
    objective = float(C.sum())
    upper_slack = 2. * width * k / (
        objective * width + 2. * k + np.sqrt((objective * width)**2 + 4. * k**2))
    return np.asarray(upper) - upper_slack


def analytic(t, mu):
    lower, upper = bounds(t)
    q = analytic_value(lower, upper, mu)
    return np.stack([q, q, np.full_like(q, .2)], axis=-1)


def independent_root(t, mu):
    """Scalar first-order condition; no call to the closed-form function."""
    lower, upper = map(float, bounds(t))
    def f(q):
        return -1. - 2. * mu / (q - lower) + 2. * mu / (upper - q)
    q = brentq(f, np.nextafter(lower, upper), np.nextafter(upper, lower),
               xtol=5e-15, rtol=4*np.finfo(float).eps)
    # Recover the multiplier from stationarity for the first flux.
    lam = C[0] + mu/(q-lower) - mu/(upper-q)
    return np.array([q, q, lam])


def kkt_rate(t, y, mu, diagnostic='complete'):
    lower, upper = bounds(t)
    lower_rate, upper_rate = bound_rates(t)
    v = y[:2]
    sl, su = v - lower, upper - v
    dl, du = mu/sl**2, mu/su**2
    rhs = dl * lower_rate + du * upper_rate
    if diagnostic == 'omit_lower':
        rhs = du * upper_rate
    elif diagnostic == 'omit_upper':
        rhs = dl * lower_rate
    K = np.block([[np.diag(dl + du), A.T], [A, np.zeros((1, 1))]])
    return np.linalg.solve(K, np.r_[rhs, 0.])


def residuals(t, values, mu):
    lower, upper = bounds(t)
    values = np.asarray(values)
    v, lam = values[:, :2], values[:, 2]
    sl, su = v-lower[:, None], upper[:, None]-v
    stationarity = -C + lam[:, None]*A.ravel() - mu/sl + mu/su
    return {'equality': np.abs(v@A.T).ravel(),
            'stationarity': np.max(np.abs(stationarity), axis=1),
            'minimum_slack': np.minimum(sl, su).min(axis=1)}


def integrate(mu, rtol, atol, diagnostic='complete', max_step=.2):
    start = time.perf_counter()
    def event(t, y):
        l, u = bounds(t)
        return min(np.min(y[:2]-l), np.min(u-y[:2])) - 1e-6
    event.terminal = True
    event.direction = -1
    solution = solve_ivp(lambda t, y: kkt_rate(t, y, mu, diagnostic),
                         (0., HORIZON), analytic(0., mu), method='DOP853',
                         rtol=rtol, atol=atol, max_step=max_step, dense_output=True,
                         events=event)
    end = float(solution.t[-1])
    grid = TIMES[TIMES <= end]
    if grid[-1] < end:
        grid = np.r_[grid, end]
    values = solution.sol(grid).T
    reference = analytic(grid, mu)
    rr = residuals(grid, values, mu)
    record = {'mu': mu, 'rtol': rtol, 'atol': atol, 'max_step_h': max_step,
              'method': 'DOP853', 'diagnostic': diagnostic,
              'solver_success': bool(solution.success), 'solver_message': solution.message,
              'terminal_time_h': end, 'reached_horizon': bool(np.isclose(end, HORIZON)),
              'interior_guard_event_time_h': solution.t_events[0].tolist(),
              'nfev': solution.nfev, 'accepted_steps': len(solution.t)-1,
              'wall_seconds': time.perf_counter()-start,
              'max_abs_flux_error': float(np.max(np.abs(values[:, :2]-reference[:, :2]))),
              'max_abs_multiplier_error': float(np.max(np.abs(values[:, 2]-reference[:, 2]))),
              'max_equality_residual': float(rr['equality'].max()),
              'max_stationarity_residual': float(rr['stationarity'].max()),
              'minimum_output_slack': float(rr['minimum_slack'].min())}
    return grid, values, reference, rr, record


def main():
    started = time.perf_counter()
    checks = []
    def check(name, passed, detail=None):
        checks.append({'check': name, 'passed': bool(passed), 'detail': detail})
    lower, upper = bounds(TIMES)
    check('Equality matrix has full row rank', np.linalg.matrix_rank(A) == 1)
    check('Analytical all-time bound-width guarantee', 1.35 - .60 > 0,
          {'lower_range': [.1, .6], 'upper_range': [1.35, 1.95], 'minimum_guaranteed_width': .75})
    reference = analytic(TIMES, MU)
    direct = np.array([independent_root(t, MU) for t in TIMES])
    rr = residuals(TIMES, reference, MU)
    check('Analytic solution has positive slacks', rr['minimum_slack'].min() > 0)
    check('Analytic equality residual', rr['equality'].max() < 1e-13)
    check('Analytic stationarity residual', rr['stationarity'].max() < 1e-12)
    direct_error = np.max(np.abs(direct-reference), axis=0)
    check('Independent scalar root agrees with closed form', direct_error.max() < 2e-12, direct_error.tolist())
    np.savetxt(OUT/'analytic_direct.csv', np.c_[TIMES, lower, upper, reference, direct],
               delimiter=',', header='time_h,lower,upper,analytic_v1,analytic_v2,analytic_lambda,direct_v1,direct_v2,direct_lambda', comments='')

    primary = integrate(MU, 1e-10, 1e-12)
    t, values, ref, res, record = primary
    check('Complete derivative reaches full horizon', record['reached_horizon'])
    check('Complete derivative retains positive output slacks', record['minimum_output_slack'] > 0)
    check('Complete derivative equality residual', record['max_equality_residual'] < 1e-12)
    check('Complete derivative flux error', record['max_abs_flux_error'] < 2e-9, record['max_abs_flux_error'])
    check('Complete derivative stationarity residual', record['max_stationarity_residual'] < 2e-8, record['max_stationarity_residual'])
    check('Complete derivative multiplier error', record['max_abs_multiplier_error'] < 1e-12)
    np.savez_compressed(OUT/'trajectory_complete.npz', time_h=t, state=values, reference=ref, **res)

    # Sensitivities are partial derivatives with the other bound held fixed.
    q = reference[:, 0]
    sl, su = q-lower, upper-q
    wl, wu = MU/sl**2, MU/su**2
    expected_l, expected_u = wl/(wl+wu), wu/(wl+wu)
    fd_rows=[]
    for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        fd_l=(analytic_value(lower+h, upper, MU)-analytic_value(lower-h, upper, MU))/(2*h)
        fd_u=(analytic_value(lower, upper+h, MU)-analytic_value(lower, upper-h, MU))/(2*h)
        fd_rows.append({'step': h, 'lower_sensitivity_max_abs_error': float(abs(fd_l-expected_l).max()),
                        'upper_sensitivity_max_abs_error': float(abs(fd_u-expected_u).max())})
    best_fd=min(fd_rows, key=lambda x:max(x['lower_sensitivity_max_abs_error'],x['upper_sensitivity_max_abs_error']))
    check('Both bound sensitivities agree with central differences',
          max(best_fd['lower_sensitivity_max_abs_error'],best_fd['upper_sensitivity_max_abs_error']) < 1e-8,best_fd)
    check('Both bound sensitivities are positive and sum to one',
          expected_l.min()>0 and expected_u.min()>0 and np.max(abs(expected_l+expected_u-1))<1e-14)
    rates=np.array([kkt_rate(t,y,MU) for t,y in zip(TIMES,reference)])
    dl,du=bound_rates(TIMES)
    chain=expected_l*dl+expected_u*du
    check('KKT rate equals partial-sensitivity chain rule',np.max(abs(rates[:,:2]-chain[:,None]))<1e-13)
    np.savetxt(OUT/'bound_sensitivities.csv',np.c_[TIMES,expected_l,expected_u,dl,du,chain],
               delimiter=',',header='time_h,dq_dlower,dq_dupper,lower_rate,upper_rate,analytic_rate',comments='')

    # Fault trajectories are stopped at a strict-interior guard, never clipped.
    faults=[]
    for diagnostic in ['omit_lower','omit_upper']:
        tt, yy, rf, rs, rec=integrate(MU,1e-10,1e-12,diagnostic)
        faults.append(rec)
        np.savez_compressed(OUT/f'trajectory_{diagnostic}.npz',time_h=tt,state=yy,reference=rf,**rs)
        check(diagnostic+' is detected as an incorrect formula',rec['max_abs_flux_error']>1e-3)
    check('Fault cases are explicitly labelled diagnostics',all(x['diagnostic']!='complete' for x in faults))

    # ODE tolerance is varied at fixed physical bounds and fixed barrier.
    tolerance_rows=[]
    for tol in [1e-5,1e-7,1e-9,1e-11]:
        _,_,_,_,rec=integrate(MU,tol,tol*.01,max_step=2.)
        tolerance_rows.append(rec)
        check('Tolerance run full horizon '+str(tol),rec['reached_horizon'])
    check('Tighter integration reduces error in this sweep',
          tolerance_rows[-1]['max_abs_flux_error']<tolerance_rows[0]['max_abs_flux_error']/100,
          [x['max_abs_flux_error'] for x in tolerance_rows])

    # Barrier refinement changes mu only: physical bounds are not relaxed.
    barrier_rows=[]
    for mu in [.2,.05,.01,.002,.0004]:
        exact=analytic(TIMES,mu)
        _,_,_,_,rec=integrate(mu,1e-10,1e-12)
        gap=upper-exact[:,0]  # c1+c2=1, optimum q=upper.
        row={'mu':mu,'physical_bound_relaxation':0.,'nominal_primal_dual_gap_bound':4*mu,
             'maximum_LP_objective_gap':float(gap.max()),'minimum_LP_objective_gap':float(gap.min()),
             'maximum_gap_divided_by_bound':float(gap.max()/(4*mu)),
             'maximum_flux_error':rec['max_abs_flux_error'],'maximum_stationarity_residual':rec['max_stationarity_residual'],
             'minimum_output_slack':rec['minimum_output_slack'],'reached_horizon':rec['reached_horizon'],
             'nfev':rec['nfev'],'wall_seconds':rec['wall_seconds']}
        barrier_rows.append(row)
        check('Central-path objective certificate '+str(mu),float(gap.min())>=0 and float(gap.max())<=4*mu*(1+1e-12))
        check('Barrier-refinement full horizon and positive slacks '+str(mu),rec['reached_horizon'] and rec['minimum_output_slack']>0)
        check('Barrier-refinement flux error '+str(mu),rec['max_abs_flux_error']<2e-8,rec['max_abs_flux_error'])
    check('Barrier refinement reduces the LP objective gap',
          all(b['maximum_LP_objective_gap']<a['maximum_LP_objective_gap'] for a,b in zip(barrier_rows,barrier_rows[1:])))
    result={'scope':'Numerical/formulation verification on a synthetic two-flux convex problem; no biological validation, solver superiority or new sensitivity-theory claim.',
            'protocol':{'horizon_h':HORIZON,'output_times':len(TIMES),'objective':'max 0.7 v1 + 0.3 v2',
                        'equality':'v1-v2=0','lower':'.35+.25 sin(.8 t)','upper':'1.65+.30 cos(.6 t+.2)',
                        'bounds_for_both_fluxes':True,'primary_mu':MU,'physical_bound_relaxation':0.,
                        'initialisation':'closed-form finite-barrier solution; identical for complete and faulty derivatives',
                        'seed':None,'seed_note':'Entire calculation is deterministic; no sampling or training.',
                        'faults':'Omitted-bound terms are deliberately incorrect diagnostic formulas, not alternative dFBA methods.',
                        'fault_stop':'Terminate at minimum slack 1e-6, before leaving the strict interior; no state clipping.',
                        'units':'Flux, objective and bound scales are nondimensional; t is a chosen synthetic time coordinate expressed in h.'},
            'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__},
            'primary':record,'independent_root_max_errors':direct_error.tolist(),
            'finite_difference_sensitivity':fd_rows,'fault_diagnostics':faults,
            'tolerance_refinement':tolerance_rows,'barrier_refinement':barrier_rows,
            'checks':checks,'checks_passed':sum(x['passed'] for x in checks),'checks_total':len(checks),
            'passed':all(x['passed'] for x in checks),'elapsed_seconds':time.perf_counter()-started,
            'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','checks_passed','checks_total','elapsed_seconds']},indent=2))
    print(json.dumps({'primary':record,'faults':faults,'barrier':barrier_rows},indent=2))
    if not result['passed']:
        raise SystemExit('\n'.join(x['check'] for x in checks if not x['passed']))


if __name__=='__main__':
    main()
