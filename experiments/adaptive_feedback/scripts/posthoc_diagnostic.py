"""Hindsight recovery scan from failed episodes' true final-interval states.

Run only AFTER the complete frozen campaign:
  python -B independent_audit/posthoc_final_interval.py --campaign LABEL
  python -B independent_audit/posthoc_final_interval.py --campaign LABEL --refine

This is an offline diagnostic, not a controller or prospective endpoint. Hidden
plant parameters and true state are used solely to inspect the last two hours
after the original actions and outcomes have been saved. A failed finite scan
does not prove physical unreachability. Different policies reach different late
states, so this does not isolate the quality of the final optimization alone.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import time
import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DT = 2.
VOLUME_CAP = 1.8
FEED_CAP = .2
ACCEPTANCE = float(np.log(1.01))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    def serial(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,default=serial,allow_nan=False)+'\n')


def state_description(state):
    state = np.asarray(state,float)
    return {'volume_L':float(state[0]),'biomass_amount_gDW':float(np.exp(state[1])+np.exp(state[2])),
            'ratio':float(np.exp(state[1]-state[2])), 'glucose_g_L':float(state[3]/state[0]),
            'remaining_volume_L':float(VOLUME_CAP-state[0]),
            'admissible_constant_feed_upper_L_h':min(FEED_CAP,max(0.,(VOLUME_CAP-state[0])/DT))}


def episode_context(row):
    stages = row.get('stages',row.get('partial_record',{}).get('completed_stages',[]))
    final_stage = next((s for s in stages if s['stage']==6),None)
    context = {'method':row['method'],'status':row['status'],
               'terminal_feasible':row['result']['terminal_feasible'],
               'terminal_ratio':row['result'].get('ratio'),
               'terminal_log_ratio_error':row['result'].get('log_ratio_error'),
               'terminal_biomass_gDW':row['result'].get('biomass_amount_gDW')}
    if final_stage:
        context.update(final_interval_true_state=state_description(final_stage['true_start_state']),
                       final_interval_actual_feed_L_h=final_stage['applied_feed_L_h'],
                       final_interval_estimated_kappa=final_stage['kappa_estimate'])
    return context


def diagnose(row, case, reactor, refine):
    started = time.perf_counter()
    call0,seconds0 = reactor.calls,reactor.seconds
    stage = row['stages'][6]
    state = np.asarray(stage['true_start_state'],float)
    actual_feed = float(stage['applied_feed_L_h'])
    true_kappa = case['kappa_after'] if 6>=case.get('change_stage',99) else case['kappa_before']
    if stage['stage']!=6 or not np.array_equal(stage['true_plant_kappa'],true_kappa):
        raise ValueError('Final stage/hidden plant schedule differs from frozen case')
    if not np.all(np.isfinite(state)) or state[0]<=0. or state[0]>VOLUME_CAP+1e-8:
        return {'case_id':case['id'],'method':row['method'],'status':'skipped',
                'reason':'Saved final-interval state is nonfinite or already violates the volume cap',
                'true_state':state.tolist()}
    upper = min(FEED_CAP,max(0.,(VOLUME_CAP-float(state[0]))/DT))
    grid = np.linspace(0.,upper,101) if upper>0. else np.array([0.])
    target_log = float(np.log(case['target_ratio']))
    cache,trace = {},[]

    def evaluate(feed, purpose):
        feed = float(feed)
        key = feed.hex()
        if key in cache:
            item = cache[key]
            if purpose not in item['purposes']: item['purposes'].append(purpose)
            return item
        start = time.perf_counter()
        try:
            q = reactor.predict(state,[feed],true_kappa,tight=True)
            error = abs(q['z']-target_log)
            valid = bool(-1e-10<=feed<=FEED_CAP+1e-10 and q['V']<=VOLUME_CAP+1e-8
                         and q['min_S']>=-2e-7 and np.all(np.isfinite(q['state'])))
            item = {'feed_L_h':feed,'status':'completed','z':q['z'],'ratio':float(np.exp(q['z'])),
                    'B_gDW':q['B'],'V_L':q['V'],'minimum_substrate_g_L':q['min_S'],
                    'terminal_state':np.asarray(q['state']).tolist(),
                    'absolute_log_ratio_error':error,'ratio_relative_error_percent':100.*abs(np.exp(q['z'])/case['target_ratio']-1.),
                    'physical_valid':valid,'terminal_feasible':bool(valid and error<=ACCEPTANCE),
                    'purposes':[purpose],'seconds':time.perf_counter()-start}
        except Exception as exc:
            item = {'feed_L_h':feed,'status':'failed','error':repr(exc),
                    'purposes':[purpose],'seconds':time.perf_counter()-start,'terminal_feasible':False}
        cache[key] = item
        trace.append(item)
        return item

    actual = evaluate(actual_feed,'actual_executed_feed_replay')
    sampled = [evaluate(feed,'uniform_grid') for feed in grid]
    valid_grid = [q for q in sampled if q['status']=='completed' and q['physical_valid']]
    best_grid = min(valid_grid,key=lambda q:q['absolute_log_ratio_error']) if valid_grid else None
    grid_feasible = [q for q in valid_grid if q['terminal_feasible']]
    refinement = {'requested':bool(refine),'performed':False,
                  'scope':'Local bounded refinement in the neighbor cell around the best grid sample; no global guarantee'}
    if refine and best_grid is not None and len(grid)>1:
        idx = int(np.argmin(np.abs(grid-best_grid['feed_L_h'])))
        lower_bracket = float(grid[max(0,idx-1)])
        upper_bracket = float(grid[min(len(grid)-1,idx+1)])
        def objective(feed):
            q = evaluate(feed,'local_scalar_refinement')
            return q['absolute_log_ratio_error'] if q['status']=='completed' and q['physical_valid'] else 1e6
        try:
            fit = minimize_scalar(objective,bounds=(lower_bracket,upper_bracket),method='bounded',
                                  options={'xatol':1e-9,'maxiter':80})
            evaluate(fit.x,'scalar_optimizer_returned_point')
            refinement.update(performed=True,success=bool(fit.success),message=str(fit.message),
                              search_bounds_L_h=[lower_bracket,upper_bracket],returned_feed_L_h=float(fit.x),
                              function_evaluations=int(fit.nfev))
        except Exception as exc:
            refinement.update(performed=True,success=False,error=repr(exc),
                              search_bounds_L_h=[lower_bracket,upper_bracket])
    alternatives = [q for q in trace if ('uniform_grid' in q['purposes'] or 'local_scalar_refinement' in q['purposes']
                    or 'scalar_optimizer_returned_point' in q['purposes']) and q['status']=='completed' and q['physical_valid']]
    best = min(alternatives,key=lambda q:q['absolute_log_ratio_error']) if alternatives else None
    feasible_found = any(q['terminal_feasible'] for q in alternatives)
    if grid_feasible:
        conclusion = 'A sampled feasible final-interval outcome exists from this saved late state under the hindsight model.'
    elif feasible_found:
        conclusion = 'A feasible final-interval outcome was found by local refinement from this saved late state under the hindsight model.'
    else:
        conclusion = 'No sampled feasible recovery from this late state; this does not prove physical unreachability.'
    if not valid_grid:
        conclusion = 'No physically valid completed grid predictions; recovery assessment is numerically inconclusive.'
    replay_difference = None
    replay_errors = None
    replay_consistent = False
    if actual['status']=='completed':
        saved_end = np.asarray(stage['true_end_state'],float)
        saved_z = float(saved_end[1]-saved_end[2])
        replay_difference = abs(actual['z']-saved_z)
        replay_errors = {'log_ratio_absolute':replay_difference,
                         'biomass_relative':abs(actual['B_gDW']/(np.exp(saved_end[1])+np.exp(saved_end[2]))-1.),
                         'volume_absolute_L':abs(actual['V_L']-saved_end[0]),
                         'glucose_amount_absolute_g':abs(actual['terminal_state'][3]-saved_end[3])}
        replay_consistent = bool(replay_difference<=2e-6 and replay_errors['biomass_relative']<=2e-6
                                 and replay_errors['volume_absolute_L']<=1e-9
                                 and replay_errors['glucose_amount_absolute_g']<=2e-6+2e-8*abs(saved_end[3]))
    prior_path_valid = bool(all(s['plant_min_S']>=-2e-7 and -1e-10<=s['applied_feed_L_h']<=.2+1e-10
                                and np.asarray(s['true_end_state'])[0]<=1.8+1e-8
                                and np.all(np.isfinite(s['true_end_state'])) for s in row['stages'][:6]))
    failure_reasons = []
    if row['result']['log_ratio_error']>ACCEPTANCE: failure_reasons.append('terminal_composition')
    if row['result']['volume_L']>1.8+1e-8: failure_reasons.append('terminal_volume')
    if not row['result']['physical_path_feed_valid']: failure_reasons.append('physical_path_or_feed')
    interpretable = bool(valid_grid and replay_consistent)
    if not replay_consistent:
        conclusion = 'Numerically inconclusive: actual-feed replay did not match the saved terminal state; alternative outcomes are provisional.'
    if not prior_path_valid:
        conclusion += ' A feasible final interval cannot erase an earlier physical-path violation.'
    return {'case_id':case['id'],'cluster':case['cluster'],'regime':case['regime'],'method':row['method'],
            'status':'completed' if interpretable else 'numerically_inconclusive',
            'diagnostic_interpretable':interpretable,'original_failure_reasons':failure_reasons,
            'preceding_path_valid':prior_path_valid,
            'target_ratio':case['target_ratio'],'saved_true_state':state,
            'late_state':state_description(state),'offline_true_kappa':true_kappa,
            'actual_feed_L_h':actual_feed,'actual_feed_within_scan_interval':bool(-1e-10<=actual_feed<=upper+1e-10),
            'actual_feed_replay':actual,'actual_feed_z_replay_error':replay_difference,
            'actual_feed_replay_errors':replay_errors,
            'actual_feed_replay_consistent':replay_consistent,
            'feed_interval_L_h':[0.,upper],'requested_grid_points':101,'unique_grid_points':len(grid),
            'zero_remaining_feed_authority':bool(upper==0.),
            'valid_grid_predictions':len(valid_grid),'failed_grid_predictions':sum(q['status']!='completed' for q in sampled),
            'feasible_grid_points':len(grid_feasible),'minimum_sampled_log_ratio_error':best_grid['absolute_log_ratio_error'] if best_grid else None,
            'best_sampled_feed_L_h':best_grid['feed_L_h'] if best_grid else None,
            'refinement':refinement,'minimum_observed_alternative_log_ratio_error':best['absolute_log_ratio_error'] if best else None,
            'best_observed_alternative_feed_L_h':best['feed_L_h'] if best else None,
            'feasible_recovery_found':feasible_found,'conclusion':conclusion,
            'evaluations':trace,'diagnostic_model_calls':reactor.calls-call0,
            'diagnostic_model_seconds':reactor.seconds-seconds0,'diagnostic_wall_seconds':time.perf_counter()-started}

