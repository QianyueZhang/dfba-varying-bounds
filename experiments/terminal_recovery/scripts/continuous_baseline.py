"""Continuous local open-loop refinement on a fixed two-step scenario tree.

The nine parameter paths are exactly K x K. No hidden process setting is read.
Two minimax SLSQP starts and, when possible, one economic refinement retain all
evaluated candidates. This is a local baseline, never a global optimum claim.
"""
from pathlib import Path
import hashlib
import time

import numpy as np
from scipy.optimize import minimize

from engine import K, flow, CAP, GUARD, BAND


INNER_MARGIN = 2e-5
COMPARISON_TOLERANCE = 1e-10
FLOW_NUMERICS = dict(rtol=2e-11, atol=2e-13, max_step_h=.05)
MAX_ITERATIONS = 60
FEED_MAX = .2
INTERVAL = 2.
SUBSTRATE_ALLOWANCE = -2e-7


def _float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _array_identity(value):
    value = np.ascontiguousarray(np.asarray(value))
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def solve_open(x, args, target, grid_plan, grid_feasible_plan=None):
    """Return a JSON-friendly best retained two-feed open-loop candidate.

    ``target`` is a positive biomass ratio. ``args`` is engine.arguments(...).
    Both supplied grid plans are retained even if infeasible. Optimizer trials
    violating capacity may be evaluated to give SLSQP a useful local function,
    but can never become the selected incumbent. All feasibility/ranking uses
    BAND - 2e-5; the outer BAND label is reported separately.
    """
    started = time.perf_counter()
    state = np.asarray(x, float).copy()
    if state.shape != (4,) or not np.all(np.isfinite(state)) or state[0] <= 0:
        raise ValueError('x must be a finite (V, logM1, logM2, Q) vector with V > 0')
    if state[0] > CAP - GUARD or state[3] / state[0] < SUBSTRATE_ALLOWANCE:
        raise ValueError('Initial state violates the declared physical domain')
    target = float(target)
    if not np.isfinite(target) or target <= 0:
        raise ValueError('target must be a positive finite ratio')
    log_target = float(np.log(target))
    inner_band = float(BAND - INNER_MARGIN)
    if inner_band <= 0:
        raise ValueError('The declared inner band must be positive')
    kappas = np.asarray(K, float).copy()
    if kappas.shape != (3, 2) or not np.all(np.isfinite(kappas)) or np.any(kappas <= 0):
        raise ValueError('This baseline requires the frozen three-scenario K')
    capacity = float(CAP - state[0] - GUARD)
    flow_count = 0
    completed_flow_count = 0
    accepted_steps = 0
    rejected_steps = 0
    evaluation_requests = 0
    cache_hits = 0
    cache = {}
    candidates = []
    solver_statuses = []
    active_origin = 'initial'

    def plan_array(plan):
        value = np.asarray(plan, float)
        if value.shape != (2,):
            raise ValueError('Each open-loop plan must contain exactly two feeds')
        return value.copy()

    def seed_projection(plan):
        projected = np.clip(plan_array(plan), 0., FEED_MAX)
        if not np.all(np.isfinite(projected)):
            return np.zeros(2)
        if INTERVAL * projected.sum() > capacity:
            projected *= capacity / (INTERVAL * projected.sum())
            while INTERVAL * projected.sum() > capacity:
                index = int(np.argmax(projected))
                projected[index] = np.nextafter(projected[index], 0.)
        return projected

    def transition(initial, feed, kappa):
        nonlocal flow_count, completed_flow_count, accepted_steps, rejected_steps
        flow_count += 1
        endpoint, minimum, steps, rejected = flow(
            initial, float(feed), kappa, args,
            FLOW_NUMERICS['rtol'], FLOW_NUMERICS['atol'], FLOW_NUMERICS['max_step_h'])
        completed_flow_count += 1
        accepted_steps += int(steps)
        rejected_steps += int(rejected)
        endpoint = np.asarray(endpoint, float)
        if not np.all(np.isfinite(endpoint)) or not np.isfinite(minimum):
            raise ValueError('Nonfinite integration output')
        return endpoint, float(minimum)

    def evaluate(plan):
        nonlocal evaluation_requests, cache_hits
        evaluation_requests += 1
        plan = plan_array(plan)
        key = plan.tobytes()
        if key in cache:
            cache_hits += 1
            record = cache[key]
            if active_origin not in record['origins']:
                record['origins'].append(active_origin)
            return record
        tick = time.perf_counter()
        record = dict(index=len(candidates), plan=[_float(v) for v in plan], origins=[active_origin],
                      numerical_valid=False, physical_valid=False, feasible=False, outer_band_feasible=False,
                      max_error=None, worst_logB=None, worstB=None, paths=[], first_states=[],
                      volume_constraint_residual_L=_float(capacity - INTERVAL * plan.sum()),
                      error=None)
        try:
            if not np.all(np.isfinite(plan)) or np.any(plan < 0) or np.any(plan > FEED_MAX):
                raise ValueError('Trial feed outside [0, 0.2] or nonfinite')
            minimum = float(state[3] / state[0])
            first = []
            for kappa in kappas:
                endpoint, stage_minimum = transition(state, plan[0], kappa)
                first.append(endpoint)
                minimum = min(minimum, stage_minimum)
                record['first_states'].append(endpoint.tolist())
            for i, endpoint in enumerate(first):
                for j, kappa in enumerate(kappas):
                    terminal, stage_minimum = transition(endpoint, plan[1], kappa)
                    minimum = min(minimum, stage_minimum)
                    z = float(terminal[1] - terminal[2])
                    log_biomass = float(np.logaddexp(terminal[1], terminal[2]))
                    if not np.isfinite(z) or not np.isfinite(log_biomass):
                        raise ValueError('Nonfinite terminal observable')
                    record['paths'].append(dict(path=[i, j], state=terminal.tolist(),
                                                z=z, logB=log_biomass,
                                                min_S_second=stage_minimum))
            record['numerical_valid'] = True
            maximum_volume = max([state[0]] + [v[0] for v in first]
                                 + [p['state'][0] for p in record['paths']])
            physical = bool(INTERVAL * plan.sum() <= capacity and maximum_volume <= CAP
                            and minimum >= SUBSTRATE_ALLOWANCE)
            max_error = max(abs(p['z'] - log_target) for p in record['paths'])
            worst_logB = min(p['logB'] for p in record['paths'])
            record.update(physical_valid=physical, max_error=float(max_error), worst_logB=worst_logB,
                          worstB=_float(np.exp(worst_logB)), minimum_substrate_g_L=minimum,
                          maximum_volume_L=float(maximum_volume),
                          feasible=bool(physical and max_error <= inner_band + COMPARISON_TOLERANCE),
                          outer_band_feasible=bool(physical and max_error <= BAND + COMPARISON_TOLERANCE))
        except Exception as exc:
            record['error'] = repr(exc)
        record['evaluation_seconds'] = time.perf_counter() - tick
        candidates.append(record)
        cache[key] = record
        return record

    def ranking(record):
        if record['feasible']:
            return (0, -record['worst_logB'], record['max_error'], record['index'])
        if record['numerical_valid'] and record['physical_valid']:
            return (1, record['max_error'], -record['worst_logB'], record['index'])
        return (2, np.inf, np.inf, record['index'])

    def composition(record):
        if not record['numerical_valid']:
            return None
        return np.asarray([p['z'] - log_target for p in record['paths']])

    active_origin = 'grid_incumbent'
    grid = plan_array(grid_plan)
    supplied_grid = evaluate(grid)
    supplied_feasible = None
    if grid_feasible_plan is not None:
        active_origin = 'grid_feasible_incumbent'
        supplied_feasible = evaluate(plan_array(grid_feasible_plan))
    uniform = min(FEED_MAX, capacity / (2 * INTERVAL))
    starts = [('grid_start', seed_projection(grid)),
              ('uniform_available_start', seed_projection([uniform, uniform]))]

    for name, seed in starts:
        active_origin = 'minimax_' + name
        initial = evaluate(seed)
        initial_slack = initial['max_error'] if initial['numerical_valid'] else 1e3

        def minimax_constraints(value):
            differences = composition(evaluate(value[:2]))
            if differences is None:
                return np.full(18, -1e6)
            return np.r_[value[2] - differences, value[2] + differences]

        try:
            fit = minimize(lambda v: float(v[2]), np.r_[seed, initial_slack], method='SLSQP',
                           bounds=[(0., FEED_MAX), (0., FEED_MAX), (0., None)],
                           constraints=[{'type': 'ineq', 'fun': minimax_constraints},
                                        {'type': 'ineq', 'fun': lambda v: capacity - INTERVAL * np.sum(v[:2])}],
                           options={'maxiter': MAX_ITERATIONS, 'ftol': 1e-10, 'eps': 1e-6})
            final = evaluate(fit.x[:2])
            solver_statuses.append(dict(stage='minimax', start=name, success=bool(fit.success),
                                        status=int(fit.status), message=str(fit.message),
                                        iterations=int(fit.nit), function_evaluations=int(fit.nfev),
                                        x=[_float(v) for v in fit.x], candidate_index=final['index']))
        except Exception as exc:
            solver_statuses.append(dict(stage='minimax', start=name, success=False, error=repr(exc)))

    feasible = [record for record in candidates if record['feasible']]
    if feasible:
        incumbent = min(feasible, key=ranking)
        active_origin = 'economic_polish'

        def economic_constraints(value):
            record = evaluate(value[:2])
            differences = composition(record)
            if differences is None:
                return np.full(27, -1e6)
            log_biomass = np.asarray([p['logB'] for p in record['paths']])
            return np.r_[inner_band - differences, inner_band + differences, log_biomass - value[2]]

        try:
            fit = minimize(lambda v: -float(v[2]), np.r_[incumbent['plan'], incumbent['worst_logB']],
                           method='SLSQP', bounds=[(0., FEED_MAX), (0., FEED_MAX), (None, None)],
                           constraints=[{'type': 'ineq', 'fun': economic_constraints},
                                        {'type': 'ineq', 'fun': lambda v: capacity - INTERVAL * np.sum(v[:2])}],
                           options={'maxiter': MAX_ITERATIONS, 'ftol': 1e-10, 'eps': 1e-6})
            final = evaluate(fit.x[:2])
            solver_statuses.append(dict(stage='economic', start_candidate_index=incumbent['index'],
                                        success=bool(fit.success), status=int(fit.status), message=str(fit.message),
                                        iterations=int(fit.nit), function_evaluations=int(fit.nfev),
                                        x=[_float(v) for v in fit.x], candidate_index=final['index']))
        except Exception as exc:
            solver_statuses.append(dict(stage='economic', start_candidate_index=incumbent['index'],
                                        success=False, error=repr(exc)))

    best = min(candidates, key=ranking)
    valid_choice = best['numerical_valid'] and best['physical_valid']
    output_plan = best['plan'] if valid_choice else seed_projection(grid).tolist()
    result = dict(plan=output_plan, best_error=best['max_error'] if valid_choice else None,
                  worstB=best['worstB'] if valid_choice else None,
                  worst_logB=best['worst_logB'] if valid_choice else None,
                  feasible=bool(valid_choice and best['feasible']),
                  outer_band_feasible=bool(valid_choice and best['outer_band_feasible']),
                  selected_candidate_index=best['index'] if valid_choice else None,
                  failure=None if valid_choice else 'No numerically valid physical candidate; returned projected supplied grid plan without a feasibility claim',
                  solver_statuses=solver_statuses,
                  count=dict(candidate_evaluations=len(candidates), evaluation_requests=evaluation_requests,
                             cache_hits=cache_hits, attempted_flow_calls=flow_count, completed_flow_calls=completed_flow_count,
                             accepted_integration_steps=accepted_steps, rejected_integration_steps=rejected_steps,
                             numerical_invalid_candidates=sum(not r['numerical_valid'] for r in candidates),
                             physical_invalid_candidates=sum(r['numerical_valid'] and not r['physical_valid'] for r in candidates)),
                  candidates=candidates,
                  verification=dict(initial_state=state.tolist(), target_ratio=target, log_target=log_target,
                                    K=kappas.tolist(), paths=[[i, j] for i in range(3) for j in range(3)],
                                    scenario_rule='Independent K choices in each interval; no hidden future input',
                                    inner_band=inner_band, outer_band=float(BAND), comparison_tolerance=COMPARISON_TOLERANCE,
                                    capacity_L=capacity, interval_h=INTERVAL, feed_bounds=[0., FEED_MAX],
                                    volume_cap_L=float(CAP), volume_guard_L=float(GUARD),
                                    substrate_allowance_g_L=SUBSTRATE_ALLOWANCE, flow_numerics=FLOW_NUMERICS,
                                    solver_options=dict(method='SLSQP', maxiter=MAX_ITERATIONS, ftol=1e-10, finite_difference_eps=1e-6),
                                    starts=[dict(name=name, plan=seed.tolist()) for name, seed in starts],
                                    original_grid_candidate_index=supplied_grid['index'],
                                    original_grid_feasible_candidate_index=None if supplied_feasible is None else supplied_feasible['index'],
                                    arguments_identity=[_array_identity(value) for value in args],
                                    family_index=int(args[0]),
                                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    ranking='Inner-band feasible first; maximum worst log biomass among feasible; otherwise minimum worst composition error among physical numerical candidates',
                                    limits='Continuous local refinement, not a global optimum. Engine numerical predictions require independent verification. Timing includes all work in this invocation and any first-use native compilation.'))
    result['seconds'] = time.perf_counter() - started
    return result
