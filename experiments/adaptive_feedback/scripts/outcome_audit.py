"""Post-campaign audit, intentionally excluded from controller timing.

Usage: python -B independent_audit/audit_outcomes.py --campaign LABEL

Checks every saved plant interval with an independently coded physical-amount
reactor and analytic volume. Checks controller tails in protocol clusters 0,3
at stages 0,3,6 with tight DOP853. No optimizer or estimator is rerun. The same
archived response functions are used, so this tests implementation, not biology
or exact unregularized FBA. Requires a finished, complete locked campaign.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time
import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DT = 2.0
DESIGN_TOL = .008
RATIO_TOL = float(np.log(1.01))
TOLERANCES = {
    'plant_z_absolute': 2e-6,
    'plant_biomass_relative': 2e-6,
    'plant_glucose_amount_absolute_g': 2e-6,
    'plant_glucose_amount_relative': 2e-8,
    'volume_absolute_L': 1e-9,
    'ordinary_tail_z_absolute': 2e-5,
    'ordinary_tail_biomass_relative': 2e-5,
    'feed_bounds_absolute_L_h': 1e-10,
    'negative_substrate_concentration_g_L': 2e-7,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def serial(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False)+'\n')


class AmountReference:
    """Separate state, rate assembly and RHS; same declared response table."""
    def __init__(self, table):
        raw = json.loads(Path(table).read_text())
        self.growth = []
        self.uptake = []
        for species in ('SC', 'KP'):
            rows = raw[species]['rows']
            nodes = np.array([row['S'] for row in rows])
            self.growth.append(PchipInterpolator(nodes, [row['growth'] for row in rows], extrapolate=False))
            self.uptake.append(PchipInterpolator(nodes, [row['glucose_flux'] for row in rows], extrapolate=False))

    def rates(self, concentration, kappa):
        substrate = max(float(concentration), 0.)
        multiplier = min(1., substrate/1e-5)
        effective = substrate*np.asarray(kappa)
        if np.any(effective > 100.) or np.any(effective < 0.):
            raise ValueError('Independent rate query outside table domain')
        g = np.array([curve(s) for curve,s in zip(self.growth,effective)])*multiplier
        u = np.array([curve(s) for curve,s in zip(self.uptake,effective)])*multiplier
        if not np.all(np.isfinite(np.r_[g,u])):
            raise ValueError('Nonfinite independently interpolated rate')
        return g,u

    def segment(self, initial, feed, kappa):
        v0 = float(initial[0])
        masses = np.r_[np.exp(initial[1:3]), initial[3]]
        def rhs(t, mass):
            volume = v0+feed*t
            g,u = self.rates(mass[2]/volume, kappa)
            return np.r_[g*mass[:2], 100.*feed+.18*np.dot(u,mass[:2])]
        sol = solve_ivp(rhs, (0.,DT), masses, method='DOP853',
                        rtol=2e-11, atol=2e-13, max_step=.05)
        if not sol.success:
            raise RuntimeError(sol.message)
        volume = v0+DT*feed
        mass = sol.y[:,-1]
        if np.any(mass[:2] <= 0):
            raise ValueError('Nonpositive biomass in independent amount replay')
        state = np.r_[volume,np.log(mass[:2]),mass[2]]
        concentration = sol.y[2]/(v0+feed*sol.t)
        return {'state':state, 'min_S':float(np.min(concentration)),
                'max_S':float(np.max(concentration)), 'rhs_calls':int(sol.nfev)}


def quantities(state):
    return {'z':float(state[1]-state[2]), 'B':float(np.exp(state[1])+np.exp(state[2])),
            'V':float(state[0]), 'Q':float(state[3])}


def state_errors(actual, reference):
    a,b = quantities(actual),quantities(reference)
    return {'z_absolute':abs(a['z']-b['z']), 'biomass_relative':abs(a['B']/b['B']-1.),
            'volume_absolute_L':abs(a['V']-b['V']), 'glucose_amount_absolute_g':abs(a['Q']-b['Q'])}


def state_error_pass(errors, reference):
    return bool(errors['z_absolute'] <= TOLERANCES['plant_z_absolute']
        and errors['biomass_relative'] <= TOLERANCES['plant_biomass_relative']
        and errors['volume_absolute_L'] <= TOLERANCES['volume_absolute_L']
        and errors['glucose_amount_absolute_g'] <= TOLERANCES['plant_glucose_amount_absolute_g']
            +TOLERANCES['plant_glucose_amount_relative']*abs(reference[3]))


def saved_trajectory_diagnostic(stage, reference, kappa):
    """Balance residual by quadrature of saved nodes, not an exact certificate."""
    times = np.asarray(stage['plant_times_local'],float)
    states = np.asarray(stage['plant_trajectory'],float)
    if len(times) != len(states) or len(times)<2 or np.any(np.diff(times)<=0):
        raise ValueError('Malformed saved plant trajectory')
    mass = np.exp(states[:,1:3])
    growth_flux, glucose_flux = [],[]
    for y,m in zip(states,mass):
        g,u = reference.rates(y[3]/y[0], kappa)
        growth_flux.append(g*m)
        glucose_flux.append(.18*np.dot(u,m))
    feed = float(stage['applied_feed_L_h'])
    biomass_integral = simpson(np.asarray(growth_flux),x=times,axis=0)
    uptake_integral = float(simpson(np.asarray(glucose_flux),x=times))
    return {
        'quadrature_note':'Simpson quadrature of adaptive saved nodes; interpolation/quadrature error included, diagnostic only',
        'biomass_balance_residual_gDW':mass[-1]-mass[0]-biomass_integral,
        'glucose_balance_residual_g':float(states[-1,3]-states[0,3]-100.*feed*(times[-1]-times[0])-uptake_integral),
        'volume_path_max_error_L':float(np.max(np.abs(states[:,0]-(states[0,0]+feed*(times-times[0]))))),
        'minimum_saved_substrate_g_L':float(np.min(states[:,3]/states[:,0])),
        'maximum_saved_substrate_g_L':float(np.max(states[:,3]/states[:,0])),
        'maximum_effective_substrate_g_L':float(np.max(states[:,3]/states[:,0])*max(kappa)),
        'saved_start_time_h':float(times[0]),
        'saved_end_time_h':float(times[-1]),
        'saved_time_span_h':float(times[-1]-times[0]),
        'saved_start_state_max_error':float(np.max(np.abs(states[0]-np.asarray(stage['true_start_state'])))),
        'saved_end_state_max_error':float(np.max(np.abs(states[-1]-np.asarray(stage['true_end_state'])))),
        'all_saved_states_finite':bool(np.all(np.isfinite(states))),
    }


def audit_episode(row, case, method, reference, tight_core):
    issues, intervals, tails = [],[],[]
    def check(condition, label):
        if not condition: issues.append(label)
    completed = row.get('status') == 'completed'
    partial = row.get('partial_record',{})
    stages = row.get('stages',partial.get('completed_stages',[]))
    check(row['case']==case and row['method']==method,'episode_identity')
    check(len(stages)==7 if completed else len(stages)<7,'number_of_stages')
    initial_m = .5*.002*np.array([case['initial_ratio'],1.])/(1.+case['initial_ratio'])
    independent_state = np.r_[.5,np.log(initial_m),.5*case['initial_glucose']]
    saved_previous = independent_state.copy()
    noise = np.asarray(case['observation_noise'])
    independent_min_s = float(case['initial_glucose'])
    independent_max_s = independent_min_s
    for j,stage in enumerate(stages):
        check(stage['stage']==j,'stage_sequence_'+str(j))
        start = np.asarray(stage['true_start_state'],float)
        end = np.asarray(stage['true_end_state'],float)
        check(np.allclose(start,saved_previous,rtol=0,atol=1e-12),'stored_continuity_'+str(j))
        plan = np.asarray(stage['plan'],float)
        feed = float(stage['applied_feed_L_h'])
        check(len(plan)==7-j,'remaining_horizon_'+str(j))
        check(abs(plan[0]-feed)<=1e-14,'first_action_'+str(j))
        check(np.all(plan>=-1e-10) and np.all(plan<=.2+1e-10),'plan_feed_bounds_'+str(j))
        check(start[0]+DT*np.sum(plan)<=1.8+1e-8,'remaining_volume_'+str(j))
        if method=='open_loop' and j>0:
            check(np.array_equal(plan,np.asarray(stages[j-1]['plan'])[1:]),'open_loop_shift_'+str(j))
        expected_observed = start.copy()
        expected_observed[1:3] += noise[j,:2]
        expected_observed[3] = max(0.,expected_observed[3]+noise[j,2]*start[0])
        check(np.allclose(stage['observed_state'],expected_observed,rtol=0,atol=2e-12),'observation_'+str(j))
        kappa = case['kappa_after'] if j>=case.get('change_stage',99) else case['kappa_before']
        check(np.array_equal(stage['true_plant_kappa'],kappa),'hidden_parameter_schedule_'+str(j))
        estimated = np.asarray(stage['kappa_estimate'])
        check(np.all(estimated>=.75-1e-12) and np.all(estimated<=1.25+1e-12),'estimator_bounds_'+str(j))
        if method.startswith('adaptive'):
            check(stage['estimation'].get('n_intervals')==min(2,j),'past_window_count_'+str(j))
        expected_clip = bool(end[3]/end[0]+noise[j+1,2]<0.)
        check(stage.get('next_substrate_observation_clipped')==expected_clip,'observation_clip_label_'+str(j))
        replay = reference.segment(independent_state,feed,kappa)
        independent_state = replay['state']
        errors = state_errors(end,independent_state)
        check(state_error_pass(errors,independent_state),'independent_amount_replay_'+str(j))
        independent_min_s = min(independent_min_s,replay['min_S'])
        independent_max_s = max(independent_max_s,replay['max_S'])
        diagnostic = saved_trajectory_diagnostic(stage,reference,kappa)
        check(diagnostic['all_saved_states_finite'],'finite_trajectory_'+str(j))
        check(diagnostic['volume_path_max_error_L']<=1e-9,'analytic_volume_path_'+str(j))
        check(abs(diagnostic['saved_time_span_h']-DT)<=1e-12,'interval_duration_'+str(j))
        check(abs(diagnostic['saved_start_time_h'])<=1e-12 and abs(diagnostic['saved_end_time_h']-DT)<=1e-12,'local_time_endpoints_'+str(j))
        check(diagnostic['saved_start_state_max_error']<=1e-12,'saved_trajectory_start_'+str(j))
        check(diagnostic['saved_end_state_max_error']<=1e-12,'saved_trajectory_end_'+str(j))
        check(abs(diagnostic['minimum_saved_substrate_g_L']-stage['plant_min_S'])<=1e-12,'saved_minimum_label_'+str(j))
        check(abs(diagnostic['maximum_saved_substrate_g_L']-stage['plant_max_S'])<=1e-12,'saved_maximum_label_'+str(j))
        check(diagnostic['minimum_saved_substrate_g_L']>=-2e-7,'saved_nonnegative_substrate_'+str(j))
        check(diagnostic['maximum_effective_substrate_g_L']<=100.+1e-10,'mapped_table_domain_'+str(j))
        intervals.append({'stage':j,'state_errors':errors,'independent_minimum_S':replay['min_S'],
                          'independent_maximum_S':replay['max_S'],'independent_rhs_calls':replay['rhs_calls'],
                          'saved_trajectory_diagnostic':diagnostic})
        if case['cluster'] in (0,3) and j in (0,3,6):
            q = tight_core.predict(np.asarray(stage['observed_state']),plan,estimated,tight=True)
            q_feasible = bool(abs(q['z']-np.log(case['target_ratio']))<=DESIGN_TOL+1e-7
                and q['V']<=1.8+1e-8 and q['min_S']>=-2e-7)
            tail = {'stage':j,'tight_z':q['z'],'tight_B':q['B'],'tight_V':q['V'],
                    'tight_minimum_S':q['min_S'],'tight_model_feasible':q_feasible,
                    'has_saved_nonlinear_prediction':stage['selected_model_z'] is not None}
            if tail['has_saved_nonlinear_prediction']:
                dz = abs(q['z']-stage['selected_model_z'])
                db = abs(q['B']/stage['selected_model_B']-1.)
                tail.update(z_absolute_error=dz,biomass_relative_error=db,
                            saved_model_feasible=stage['selected_model_feasible'],
                            feasibility_label_agrees=q_feasible==stage['selected_model_feasible'],
                            distance_to_design_band=abs(abs(q['z']-np.log(case['target_ratio']))-DESIGN_TOL))
                check(dz<=TOLERANCES['ordinary_tail_z_absolute'],'ordinary_tail_z_'+str(j))
                check(db<=TOLERANCES['ordinary_tail_biomass_relative'],'ordinary_tail_biomass_'+str(j))
                if not tail['feasibility_label_agrees']:
                    issues.append('model_feasibility_boundary_disagreement_'+str(j))
            else:
                tail['note'] = 'Post hoc nonlinear diagnostic; this method/stage did not save a selected nonlinear prediction'
            tails.append(tail)
        saved_previous = end
    total_controller = float(row.get('controller_seconds',0.))
    stage_seconds = sum(float(s['controller_seconds']) for s in stages)
    stage_calls = sum(int(s['model_calls']) for s in stages)
    if completed:
        check(abs(total_controller-stage_seconds)<=1e-8,'complete_controller_time_sum')
        check(stage_calls==row['model_calls'],'complete_model_call_sum')
        check(row['estimator_seconds']<=total_controller+1e-8,'estimator_is_subset_of_controller_time')
        check(row['model_seconds']<=total_controller+1e-8,'model_time_is_subset_of_controller_time')
        check(np.allclose(np.asarray(row['true_states']),np.vstack([np.r_[.5,np.log(initial_m),.5*case['initial_glucose']]]+[np.asarray(s['true_end_state']) for s in stages]),rtol=0,atol=1e-12),'episode_state_list')
        check(np.array_equal(row['controls'],[s['applied_feed_L_h'] for s in stages]),'episode_action_list')
        saved_end = np.asarray(stages[-1]['true_end_state'],float)
        saved_q = quantities(saved_end)
        saved_ratio = float(np.exp(saved_q['z']))
        saved_log_error = abs(saved_q['z']-np.log(case['target_ratio']))
        expected_result = {
            'volume_L':saved_q['V'],'biomass_amount_gDW':saved_q['B'],
            'log_ratio':saved_q['z'],'ratio':saved_ratio,'glucose_g_L':saved_q['Q']/saved_q['V'],
            'SC_gDW_L':float(np.exp(saved_end[1])/saved_q['V']),
            'KP_gDW_L':float(np.exp(saved_end[2])/saved_q['V']),
            'log_ratio_error':saved_log_error,
            'ratio_relative_error_percent':100.*abs(saved_ratio/case['target_ratio']-1.),
            'terminal_ratio_violation':max(0.,saved_log_error-RATIO_TOL),
            'minimum_substrate':min(s['plant_min_S'] for s in stages),
            'maximum_substrate':max(s['plant_max_S'] for s in stages),
        }
        for field,value in expected_result.items():
            check(np.isclose(row['result'].get(field,float('nan')),value,rtol=1e-12,atol=1e-12),'reported_numeric_result_'+field)
        end = quantities(independent_state)
        independent_terminal = bool(abs(end['z']-np.log(case['target_ratio']))<=RATIO_TOL and end['V']<=1.8+1e-8)
        independent_physical = bool(independent_min_s>=-2e-7 and all(-1e-10<=s['applied_feed_L_h']<=.2+1e-10 for s in stages)
                                    and end['V']<=1.8+1e-8 and np.all(np.isfinite(independent_state)))
        check(row['result']['terminal_ratio_volume_feasible']==independent_terminal,'terminal_ratio_volume_classification')
        check(row['result']['physical_path_feed_valid']==independent_physical,'physical_classification')
        check(row['result']['terminal_feasible']==(independent_terminal and independent_physical),'combined_classification')
        terminal = {'independent_z':end['z'],'independent_B':end['B'],'independent_V':end['V'],
                    'independent_minimum_S':independent_min_s,'independent_maximum_S':independent_max_s,
                    'independent_terminal_ratio_volume_feasible':independent_terminal,
                    'independent_physical_valid':independent_physical,
                    'distance_to_acceptance_band':abs(abs(end['z']-np.log(case['target_ratio']))-RATIO_TOL)}
    else:
        check(not row['result']['terminal_feasible'],'failed_episode_counted_as_failure')
        check(total_controller+1e-8>=stage_seconds,'failed_attempt_time_includes_completed_stages')
        terminal = {'status':'failed_episode','saved_error':row.get('error'),
                    'completed_stages':len(stages),'failed_stage_controller_seconds':total_controller-stage_seconds}
    return {'case_id':case['id'],'method':method,'status':row.get('status'),
            'issues':issues,'passed':not issues,'intervals':intervals,'controller_tail_checks':tails,
            'terminal':terminal,'cost_check':{'controller_seconds':total_controller,
                'completed_stage_controller_seconds':stage_seconds,'completed_stage_model_calls':stage_calls}}

