"""Post hoc transport of saved causal forecast policies to the TRUE plant.

DO NOT run during the main campaign or serial timing panel. This is offline
counterfactual replay, not the actual replanning controller and not an estimate
of a false-accept causal effect. No controller or study outputs are changed.

Usage after execution authorization:
 python -B audit/forecast_transport.py --results results/locked --output audit/forecast_transport_results.json
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import platform
import sys
import time
import numpy as np
import scipy
from independent_amount_model import AmountReactor, from_log, to_log, DT, CAP, FEED_MAX, GUARD, RTOL, ATOL, MAX_STEP

HERE=Path(__file__).resolve().parent
PROTOCOL_PATH=HERE/'forecast_transport_protocol.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,default=serial,allow_nan=False)+'\n')
    temp.replace(path)


def observed(log_state,noise):
    """Same measurement equation, applied to this replay's own physical state."""
    y=np.asarray(log_state,float).copy();error=np.asarray(noise,float)
    y[1:3]+=error[:2]
    y[3]=max(0.,y[3]+error[2]*y[0])
    return y


def realized_outcome(row):
    r=row.get('result',{})
    return {'episode_status':row.get('status'),'terminal_feasible':r.get('terminal_feasible'),
        'operational_success':r.get('operational_success'),'physical_valid':r.get('physical_valid'),
        'controller_failure':r.get('controller_failure'),'log_ratio':r.get('log_ratio'),
        'log_ratio_error':r.get('log_ratio_error'),'biomass_amount_gDW':r.get('biomass_amount_gDW')}


def model_summary(decision,target,settings):
    selected=decision['selected'];z=np.asarray(selected['z'],float)
    if z.ndim!=1 or not len(z) or not np.isfinite(z).all():raise ValueError('Invalid saved finite-scenario predictions')
    names=[s['name'] for s in decision['scenarios']]
    if len(names)!=len(z):raise ValueError('Saved scenario names/predictions differ in length')
    if 'center' not in names:raise ValueError('No saved center scenario')
    error=float(np.max(np.abs(z-target)))
    return {'saved_selected_feasible':bool(selected['feasible']),
        'saved_inner_label_recomputed':error<=settings['inner_optimization_band']+settings['inner_gate_slack'],
        'saved_outer_design_feasible':error<=settings['design_log_ratio_tolerance'],
        'saved_maximum_target_error':error,'scenario_names':names,'scenario_terminal_z':z,
        'scenario_min_z':float(z.min()),'scenario_max_z':float(z.max()),'center_z':float(z[names.index('center')]),
        'managed_gate_accepted':any(e.get('action')=='accept' for e in decision.get('events',[])),
        'fallback_used':bool(decision.get('fallback',False)),
        'optimizer_status':decision.get('status')}


def replay_one(row,stage,case,table,settings,record):
    """Mutates record progressively so a failed replay retains its full prefix."""
    start=int(stage['stage']);n=settings['horizon_intervals']-start;decision=stage['decision']
    if 'controller_exception' in decision:
        record.update(status='no_saved_policy',reason='controller exception',controller_exception=decision['controller_exception'])
        return
    target=float(np.log(case['target_ratio']))
    record['model_forecast']=model_summary(decision,target,settings)
    theta=np.asarray(decision['theta'],float);reference=np.asarray(decision['reference'],float);basis=np.asarray(decision['basis'],float)
    if theta.shape!=(n+1,) or reference.shape!=(n,) or basis.shape!=(n,):raise ValueError('Policy horizon/array mismatch')
    if not all(np.isfinite(v).all() for v in (theta,reference,basis)):raise ValueError('Nonfinite policy arrays')
    if not (0<=theta[-1]<=2 and np.all((theta[:-1]>=0)&(theta[:-1]<=settings['feed_max_L_h']))):raise ValueError('Saved policy parameter bounds violated')
    noises=np.asarray(case.get('noise',np.zeros((8,3))),float)
    if noises.shape!=(8,3) or not np.isfinite(noises).all():raise ValueError('Invalid fixed observation-noise array')
    y=from_log(stage['true_start_state']);first_observed=observed(to_log(y),noises[start])
    observation_error=float(np.max(np.abs(first_observed-np.asarray(stage['observed_state']))))
    if observation_error>settings['saved_arithmetic_atol']:raise ValueError('Saved starting observation does not match true start plus case noise')
    plant=AmountReactor(case.get('family','table'),case.get('inhibition'),table=table)
    physical=bool(y[0]>0 and y[0]<=settings['volume_max_L'] and y[1]>0 and y[2]>0 and y[3]/y[0]>=settings['substrate_min_g_L'])
    minimum=float(y[3]/y[0]);record['counterfactual_trajectory']=[]
    record['policy']={'theta':theta,'reference':reference,'basis':basis}
    record['initial_true_amount_state']=y.copy();record['initial_observation_identity_error']=observation_error
    for local in range(n):
        global_stage=start+local
        own_true_log=to_log(y);measurement=observed(own_true_log,noises[global_stage])
        z_observed=float(measurement[1]-measurement[2]);raw=float(theta[local])
        if local>0:raw-=theta[-1]*basis[local]*(z_observed-reference[local])
        # Volume observation is exact; the guard uses this same physical V.
        if measurement[0]!=y[0]:raise AssertionError('Volume observation unexpectedly changed')
        cap=max(0.,min(settings['feed_max_L_h'],(settings['volume_max_L']-measurement[0]-settings['volume_guard_L'])/settings['interval_h']))
        if cap<=0:feed=0.;branch='exhausted'
        elif raw<0:feed=0.;branch='lower'
        elif raw>cap:feed=cap;branch='volume' if cap<settings['feed_max_L_h'] else 'upper'
        else:feed=raw;branch='interior'
        if local==0:
            gap=abs(feed-float(stage['feed']));record['first_action_difference_from_realized']=gap
            if gap>settings['saved_arithmetic_atol']:raise ValueError('Transported first action differs from saved realized action')
        k=case['kappa_after'] if global_stage>=case.get('change_stage',99) else case['kappa_before']
        item={'local_interval':local,'global_stage':global_stage,'true_start_amount_state':y.copy(),
            'own_observed_state':measurement,'observation_noise':noises[global_stage],
            'observed_log_ratio':z_observed,'raw_command':raw,'guard_cap':cap,'feed':feed,'guard_branch':branch,
            'true_kappa_offline_only':k}
        record['counterfactual_trajectory'].append(item)
        result=plant.interval(y,feed,k);y=result['amount_state'];minimum=min(minimum,result['min_S'])
        physical &= bool(0<=feed<=settings['feed_max_L_h'] and y[0]<=settings['volume_max_L'] and result['min_S']>=settings['substrate_min_g_L'])
        item.update(true_end_amount_state=y.copy(),minimum_substrate=result['min_S'],maximum_substrate=result['max_S'])
    z=float(np.log(y[1]/y[2]));error=abs(z-target);model=record['model_forecast']
    lower=model['scenario_min_z'];upper=model['scenario_max_z'];cushion=settings['forecast_envelope_numerical_cushion']
    outside=z-lower if z<lower else z-upper if z>upper else 0.
    terminal=bool(physical and error<=settings['terminal_log_ratio_tolerance'])
    record.update(status='completed',counterfactual={'terminal_z':z,'target_log_ratio':target,'target_error':error,
        'physical_valid':physical,'terminal_feasible':terminal,'design_band_feasible':bool(physical and error<=settings['design_log_ratio_tolerance']),
        'terminal_amount_state':y,'terminal_log_state':to_log(y),'biomass_amount_gDW':float(y[1:3].sum()),'minimum_substrate':minimum,
        'inside_finite_forecast_envelope':bool(lower<=z<=upper),
        'inside_numerically_padded_forecast_envelope':bool(lower-cushion<=z<=upper+cushion),
        'signed_distance_outside_forecast_envelope':float(outside),'distance_outside_forecast_envelope':float(abs(outside)),
        'true_minus_center_terminal_z':z-model['center_z'],
        'model_feasible_counterfactual_failed':bool(model['saved_selected_feasible'] and not terminal)})


def summarize(records):
    complete=[r for r in records if r['status']=='completed']
    model_known=[r for r in records if 'model_forecast' in r]
    model_feasible=[r for r in model_known if r['model_forecast']['saved_selected_feasible']]
    mf_complete=[r for r in complete if r['model_forecast']['saved_selected_feasible']]
    counterfailed=[r for r in mf_complete if r['counterfactual']['model_feasible_counterfactual_failed']]
    exact=sum(r['counterfactual']['inside_finite_forecast_envelope'] for r in complete)
    padded=sum(r['counterfactual']['inside_numerically_padded_forecast_envelope'] for r in complete)
    def stats(values):
        return {'n':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),'maximum':float(np.max(values))} if values else {'n':0,'mean':None,'median':None,'maximum':None}
    return {'planned_policy_slots':len(records),'status_counts':dict(collections.Counter(r['status'] for r in records)),
        'completed_replays':len(complete),'unknown_or_unavailable_replays':len(records)-len(complete),
        'saved_model_feasible_known':len(model_feasible),'completed_model_feasible':len(mf_complete),
        'model_feasible_unknown_or_unavailable':len(model_feasible)-len(mf_complete),
        'model_feasible_counterfactual_failed':len(counterfailed),
        'model_feasible_counterfactual_design_band_failed':sum(not r['counterfactual']['design_band_feasible'] for r in mf_complete),
        'model_feasible_counterfactual_failed_fraction':len(counterfailed)/len(mf_complete) if mf_complete else None,
        'finite_envelope_contains_count':exact,'finite_envelope_contains_fraction':exact/len(complete) if complete else None,
        'padded_envelope_contains_count':padded,'padded_envelope_contains_fraction':padded/len(complete) if complete else None,
        'counterfactual_terminal_feasible_count':sum(r['counterfactual']['terminal_feasible'] for r in complete),
        'counterfactual_physical_failure_count':sum(not r['counterfactual']['physical_valid'] for r in complete),
        'absolute_center_z_discrepancy':stats([abs(r['counterfactual']['true_minus_center_terminal_z']) for r in complete]),
        'distance_outside_forecast_envelope':stats([r['counterfactual']['distance_outside_forecast_envelope'] for r in complete]),
        'counterfactual_target_error':stats([r['counterfactual']['target_error'] for r in complete]),
        'saved_inner_label_discrepancy_count':sum(r['model_forecast']['saved_selected_feasible']!=r['model_forecast']['saved_inner_label_recomputed'] for r in model_known)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier diagnostic outputs; choose a new output path')
    started=time.perf_counter();protocol=json.loads(PROTOCOL_PATH.read_text());settings=protocol['physical_and_numerical_settings'];selection=protocol['selection']
    result_dir=args.results.resolve();snapshot=result_dir/'source_snapshot';manifest_path=result_dir/'execution.json'
    manifest=json.loads(manifest_path.read_text());cases=manifest['cases'];table=snapshot/'flux_table_eps1e-06.json';issues=[]
    settings_match=(DT,CAP,FEED_MAX,GUARD,RTOL,ATOL,MAX_STEP)==(settings['interval_h'],settings['volume_max_L'],settings['feed_max_L_h'],settings['volume_guard_L'],settings['DOP853_rtol'],settings['DOP853_atol'],settings['max_step_h'])
    if not settings_match:issues.append('Independent numerical/physical settings differ from diagnostic protocol')
    if manifest.get('mode')!='evaluation':issues.append('Input is not a frozen evaluation manifest')
    if not (result_dir/'complete.json').exists():issues.append('Main campaign completion record absent; do not run during campaign')
    original_protocol=snapshot/'protocol.json'
    if not original_protocol.exists():issues.append('Saved frozen study protocol absent')
    else:
        study=json.loads(original_protocol.read_text())
        if study['cases']!=cases:issues.append('Manifest is not the full original case list; selection ordinals would differ')
    for relative in ('data/flux_table_eps1e-06.json','audit/independent_amount_model.py'):
        expected=manifest['hashes'].get(relative);saved=snapshot/Path(relative).name
        if not expected or not saved.exists() or sha(saved)!=expected:issues.append('Source snapshot identity failed: '+relative)
        if relative.startswith('audit/') and expected and sha(HERE/Path(relative).name)!=expected:issues.append('Current independent amount model differs from frozen campaign source')
    selected_cases=[(i,c) for i,c in enumerate(cases) if i%selection['case_index_modulus']==selection['case_index_remainder']]
    records=[];episode_inventory=[]
    for i,case in selected_cases:
        for method in selection['methods']:
            file=result_dir/f'{case["id"]}__{method}.json';row=None;error=None
            if issues:error='; '.join(issues)
            elif method not in manifest['methods']:error='Requested method absent from assigned manifest'
            elif not file.exists():error='Assigned episode file absent'
            else:
                try:
                    row=json.loads(file.read_text())
                    if row.get('case')!=case or row.get('method')!=method or row.get('source_hashes')!=manifest['hashes']:
                        error='Episode case/method/source identity mismatch'
                except Exception as ex:
                    error=repr(ex)
                    if not isinstance(row,dict):row=None
            episode_inventory.append({'case_index':i,'case':case['id'],'method':method,'file':file.name,
                'sha256':sha(file) if file.exists() else None,'load_issue':error,'saved_actual_outcome':realized_outcome(row) if row else None})
            stages=[] if row is None else row.get('stages',[]) if row.get('status')=='completed' else row.get('partial',{}).get('completed_stages',[])
            indexed={s['stage']:s for s in stages}
            for stage_index in selection['stages']:
                record={'case_index':i,'case':case['id'],'method':method,'stage':stage_index,'family':case.get('family','table'),
                    'regime':case.get('regime'),'cluster':case.get('cluster'),'status':'pending',
                    'saved_actual_outcome':realized_outcome(row) if row else None}
                records.append(record)
                if error:record.update(status='input_unavailable_or_invalid',reason=error);continue
                if stage_index not in indexed:record.update(status='saved_stage_unavailable',reason='Episode has no saved completed stage at this ordinal');continue
                try:replay_one(row,indexed[stage_index],case,table,settings,record)
                except Exception as ex:record.update(status='replay_failed',reason=repr(ex))
    groups={}
    for key in ('method','stage','family','regime'):
        values=sorted(set(str(r[key]) for r in records))
        groups[key]={value:summarize([r for r in records if str(r[key])==value]) for value in values}
    actual_summary={}
    for method in selection['methods']:
        episodes=[e for e in episode_inventory if e['method']==method]
        actual=[e['saved_actual_outcome'] for e in episodes if e['load_issue'] is None and e['saved_actual_outcome'] is not None]
        actual_summary[method]={'assigned_selected_episodes':len(episodes),'available_episode_records':len(actual),
            'completed_episode_records':sum(a['episode_status']=='completed' for a in actual),
            'saved_terminal_feasible_count':sum(a['terminal_feasible'] is True for a in actual),
            'saved_operational_success_count':sum(a['operational_success'] is True for a in actual),
            'note':'Each episode counted once; these are original realized replanning outcomes, separate from policy-stage counterfactuals.'}
    report={'diagnostic':'Post hoc counterfactual forecast transport','posthoc':True,'controller_tuning':False,
        'protocol':protocol,'protocol_sha256':sha(PROTOCOL_PATH),'source_manifest':str(manifest_path),'source_manifest_sha256':sha(manifest_path),
        'diagnostic_source_sha256':{p.name:sha(p) for p in (Path(__file__),HERE/'independent_amount_model.py')},
        'input_issues':issues,'summary':summarize(records),'grouped_summaries':groups,'saved_realized_episode_summary':actual_summary,
        'selected_episodes':episode_inventory,'policy_records':records,
        'elapsed_seconds':time.perf_counter()-started,'runtime':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
        'interpretation':'A fixed saved policy is transported to the hidden simulated plant and its own noisy observations. Actual deployment replans. These counterfactual discrepancies are not realized false accepts or evidence of gate causality.'}
    dump(args.output,report)
    print(json.dumps({'output':str(args.output.resolve()),'input_issues':issues,'summary':report['summary'],'elapsed_seconds':report['elapsed_seconds']},indent=2,allow_nan=False))


if __name__=='__main__':main()
