"""Independent audit of frozen saved campaigns; never imports deployed model code.

Usage: python -B audit/replay_campaign.py --results results/LABEL
Thresholds and deterministic tail subset are in campaign_audit_protocol.json.
The main chained replay never resets a physical state to a saved intermediate
state. Prediction tails intentionally start from their saved observed state,
using the saved frozen reference, basis, theta and scenario paths verbatim.
"""
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
from independent_amount_model import AmountReactor,initial_amounts,from_log,to_log,RTOL,ATOL,MAX_STEP,DT,CAP,GUARD
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PROTOCOL_PATH=HERE/'campaign_audit_protocol.json'
P=json.loads(PROTOCOL_PATH.read_text());T=P['tolerances'];C=P['physical_constraints']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,default=serial)+'\n')
def close(a,b,atol,rtol=0):return bool(np.allclose(a,b,atol=atol,rtol=rtol))
def difference(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


class Audit:
    def __init__(self):self.failures=[];self.qualifications=[];self.maxima={}
    def check(self,condition,code,context=None,detail=None,qualification=False):
        if not condition:
            (self.qualifications if qualification else self.failures).append({'code':code,'context':context,'detail':detail})
    def max(self,name,value):self.maxima[name]=max(self.maxima.get(name,0.),float(value))


def check_state(audit,expected,saved,context):
    """Expected in physical amounts, saved in original log-state coordinates."""
    saved=np.asarray(saved,float);actual=to_log(expected);saved_amount=from_log(saved)
    for index,name in ((0,'volume'),(1,'log_SC'),(2,'log_KP'),(3,'glucose_amount')):
        audit.max('actual_'+name+'_absolute_error',abs(actual[index]-saved[index]))
    audit.check(abs(actual[0]-saved[0])<=T['volume_replay_atol_L'],'state_volume',context,{'expected':actual[0],'saved':saved[0]})
    audit.check(close(actual[1:3],saved[1:3],T['log_state_atol']),'state_log_biomass',context,{'error':difference(actual[1:3],saved[1:3])})
    audit.check(close(expected[1:],saved_amount[1:],T['amount_atol_g'],T['amount_rtol']),'state_physical_amounts',context,{'errors':(expected-saved_amount).tolist()})


def check_observation(audit,true_saved,observed_saved,noise,context):
    expected=np.asarray(true_saved,float).copy();noise=np.asarray(noise,float)
    expected[1:3]+=noise[:2];expected[3]=max(0.,expected[3]+noise[2]*expected[0])
    audit.check(close(expected,observed_saved,T['saved_arithmetic_atol']),'observation_noise_identity',context,{'error':difference(expected,observed_saved)})


def replay_tail(audit,row,stage,table,context):
    decision=stage.get('decision',{})
    if 'controller_exception' in decision:return {'skipped':'controller exception has no selected policy'}
    required=['theta','reference','basis','scenarios','selected']
    if any(k not in decision for k in required):
        audit.check(False,'tail_missing_policy',context,required);return {'skipped':'missing policy fields'}
    model=AmountReactor(row['case'].get('family','table'),table=table)
    y=from_log(stage['observed_state']);theta=decision['theta'];reference=decision['reference'];basis=decision['basis'];selected=decision['selected']
    scenarios=decision['scenarios'];saved_rows=selected['rows'];rows=[];target=np.log(row['case']['target_ratio'])
    audit.check(len(scenarios)==len(saved_rows),'tail_scenario_row_count',context)
    for i,scenario in enumerate(scenarios):
        sub=dict(context,scenario=scenario['name']);q=model.policy(y,theta,scenario['kappas'],reference,basis,False)
        old=saved_rows[i]
        for field in ('z','logB'):
            error=abs(q[field]-old[field]);audit.max('policy_'+field+'_absolute_error',error)
            audit.check(error<=T['policy_z_logB_atol'],'tail_'+field,sub,{'expected':q[field],'saved':old[field],'error':error})
        feed_error=difference(q['feeds'],old['feeds']);audit.max('policy_feed_absolute_error',feed_error)
        audit.check(feed_error<=T['policy_feed_atol'],'tail_feeds',sub,{'error':feed_error})
        state_error=difference(q['state'],old['state']);audit.max('policy_state_absolute_error',state_error)
        audit.check(state_error<=T['policy_state_atol'],'tail_state',sub,{'error':state_error})
        audit.check(np.all((q['feeds']>=0)&(q['feeds']<=C['feed_max_L_h'])) and q['V']<=C['volume_max_L'] and q['min_S']>=C['substrate_min_g_L'],'tail_physical',sub,{'V':q['V'],'min_S':q['min_S']})
        audit.check(close(np.asarray(old['knots'])[0],stage['observed_state'],T['saved_arithmetic_atol']),'tail_start_observation',sub)
        audit.check(abs(old['z']-selected['z'][i])<=T['saved_arithmetic_atol'],'tail_saved_z_array',sub)
        audit.check(abs(old['logB']-selected['logB'][i])<=T['saved_arithmetic_atol'],'tail_saved_logB_array',sub)
        rows.append({'scenario':scenario['name'],'z':q['z'],'logB':q['logB'],'V':q['V'],'min_S':q['min_S'],'feeds':q['feeds'],'branches':q['branches'],
                     'outer_design_feasible':abs(q['z']-target)<=C['design_log_ratio_tolerance']})
    error=max(abs(q['z']-target) for q in rows)
    inner=error<=C['inner_optimization_band']+C['inner_gate_slack'];outer=error<=C['design_log_ratio_tolerance']
    audit.check(inner==selected['feasible'],'inner_gate_label',context,{'independent':inner,'saved':selected['feasible'],'max_error':error,'distance_to_inner_boundary':error-C['inner_optimization_band']},qualification=True)
    if selected['feasible']:
        audit.check(outer,'accepted_design_band',context,{'max_error':error,'design_band':C['design_log_ratio_tolerance']})
    first=[r['feeds'][0] for r in rows]
    audit.check(difference(first,np.full(len(first),first[0]))<=T['saved_arithmetic_atol'],'scenario_first_action_common',context)
    audit.check(abs(first[0]-stage['requested_feed'])<=T['policy_feed_atol'],'selected_requested_feed',context)
    return {'rows':rows,'inner_gate_feasible':inner,'outer_design_feasible':outer,'maximum_log_ratio_error':error}


def replay_episode(audit,row,table,manifest_case,manifest_method,tail_selected):
    case=row.get('case',{});context={'case':case.get('id'),'method':row.get('method')}
    audit.check(case==manifest_case,'episode_case_identity',context)
    audit.check(row.get('method')==manifest_method,'episode_method_identity',context)
    completed=row.get('status')=='completed'
    stages=row.get('stages',[]) if completed else row.get('partial',{}).get('completed_stages',[])
    audit.check(len(stages)==7 if completed else len(stages)<8,'episode_stage_count',context,{'completed':completed,'n':len(stages)})
    y=initial_amounts(case['initial_ratio'],case['initial_glucose']);plant=AmountReactor(case.get('family','table'),case.get('inhibition'),table)
    noises=np.asarray(case.get('noise',np.zeros((8,3))),float);physical=True;minimum=float(y[3]/y[0]);records=[];tails=[];controller_failed=False
    if completed and row.get('states'):
        audit.check(len(row['states'])==8,'saved_states_length',context)
        check_state(audit,y,row['states'][0],dict(context,stage='initial'))
    for j,stage in enumerate(stages):
        sub=dict(context,stage=j);audit.check(stage['stage']==j,'stage_index',sub)
        check_state(audit,y,stage['true_start_state'],dict(sub,anchor='start'))
        check_observation(audit,stage['true_start_state'],stage['observed_state'],noises[j],sub)
        k=case['kappa_after'] if j>=case.get('change_stage',99) else case['kappa_before']
        audit.check(np.array_equal(stage['true_kappa'],k),'true_kappa_case_identity',sub)
        feed=float(stage['feed']);requested=float(stage['requested_feed'])
        audit.check(np.isfinite(feed) and 0<=feed<=C['feed_max_L_h'],'exact_feed_bounds',sub,feed)
        expected_feed=max(0.,min(C['feed_max_L_h'],requested,(C['volume_max_L']-stage['true_start_state'][0]-C['volume_guard_L'])/C['interval_h']))
        audit.check(abs(feed-expected_feed)<=T['saved_arithmetic_atol'],'executed_feed_guard',sub,{'feed':feed,'expected':expected_feed})
        audit.check(abs((feed-requested)-stage['projection_change'])<=T['saved_arithmetic_atol'],'projection_change_bookkeeping',sub)
        endpoint=plant.interval(y,feed,k);y=endpoint['amount_state'];minimum=min(minimum,endpoint['min_S'])
        physical &= bool(0<=feed<=C['feed_max_L_h'] and y[0]<=C['volume_max_L'] and endpoint['min_S']>=C['substrate_min_g_L'])
        audit.check(stage['true_end_state'][0]<=C['volume_max_L'],'saved_exact_volume_cap',sub,stage['true_end_state'][0])
        analytic_saved_volume=stage['true_start_state'][0]+C['interval_h']*feed
        audit.check(abs(stage['true_end_state'][0]-analytic_saved_volume)<=T['saved_arithmetic_atol'],'saved_volume_balance',sub)
        check_state(audit,y,stage['true_end_state'],dict(sub,anchor='end'))
        if completed:check_state(audit,y,row['states'][j+1],dict(sub,anchor='states_array'))
        controller_failed |= 'controller_exception' in stage.get('decision',{})
        if tail_selected and j in P['selected_tail_subset']['stages']:
            tails.append({'stage':j,**replay_tail(audit,row,stage,table,sub)})
        records.append({'stage':j,'amount_state':y.copy(),'state':to_log(y),'feed':feed,'true_kappa':k,'min_S':endpoint['min_S'],'max_S':endpoint['max_S']})
    z=float(np.log(y[1]/y[2]));error=abs(z-np.log(case['target_ratio']));terminal=bool(completed and physical and error<=C['terminal_log_ratio_tolerance']);operational=bool(terminal and not controller_failed)
    if completed:
        result=row['result']
        for key,expected in [('terminal_feasible',terminal),('operational_success',operational),('physical_valid',physical),('controller_failure',controller_failed)]:
            audit.check(result.get(key)==expected,'actual_'+key,context,{'expected':expected,'saved':result.get(key),'terminal_error':error})
        expected_values={'volume_L':y[0],'biomass_amount_gDW':y[1:3].sum(),'log_ratio':z,'ratio':y[1]/y[2],
          'glucose_g_L':y[3]/y[0],'SC_gDW_L':y[1]/y[0],'KP_gDW_L':y[2]/y[0],'log_ratio_error':error}
        for key,expected in expected_values.items():
            tol=T['log_state_atol'] if key in ('log_ratio','log_ratio_error') else T['amount_atol_g']
            audit.check(close(result[key],expected,tol,T['amount_rtol']),'result_'+key,context,{'saved':result[key],'expected':expected})
        audit.check(abs(sum(s['controller_seconds'] for s in stages)-row['controller_seconds'])<=T['cost_sum_atol_s'],'controller_cost_sum',context)
        audit.check(sum(s['model_calls'] for s in stages)==row['model_calls'],'model_call_sum',context,{'stages_sum':sum(s['model_calls'] for s in stages),'total':row['model_calls']})
    else:
        audit.check(not row.get('result',{}).get('terminal_feasible',False) and not row.get('result',{}).get('operational_success',False),'failed_episode_not_success',context)
        audit.check(False,'retained_failed_episode',context,row.get('error'),qualification=True)
    return dict(context,completed=completed,completed_stages=len(stages),physical_valid=physical,terminal_feasible=terminal,operational_success=operational,
                log_ratio_error=error,biomass_amount_gDW=float(y[1:3].sum()),minimum_substrate=minimum,trajectory=records,selected_tails=tails)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',required=True,type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--development-smoke',action='store_true',help='Allows partial pilot manifests; report is explicitly not a frozen-campaign audit.')
    args=parser.parse_args();result_dir=args.results.resolve();output=args.output or result_dir/'independent_amount_audit.json'
    audit=Audit();started=time.perf_counter();manifest=json.loads((result_dir/'execution.json').read_text());snapshot=result_dir/'source_snapshot'
    audit.check((RTOL,ATOL,MAX_STEP)==(P['numerics']['rtol'],P['numerics']['atol'],P['numerics']['max_step_h']),'independent_numerical_protocol')
    audit.check((DT,CAP,GUARD)==(C['interval_h'],C['volume_max_L'],C['volume_guard_L']),'independent_physical_protocol')
    if not args.development_smoke:audit.check(manifest['mode']=='evaluation','frozen_evaluation_mode')
    hashes=manifest['hashes']
    for relative,expected in hashes.items():
        path=snapshot/Path(relative).name
        audit.check(path.exists() and digest(path)==expected,'snapshot_source_identity',{'file':relative})
    table=snapshot/'flux_table_eps1e-06.json'
    audit.check('data/flux_table_eps1e-06.json' in hashes,'source_table_declared')
    if any('import kernel' in (snapshot/Path(p).name).read_text() for p in hashes if p.endswith('.py')):
        audit.check('scripts/kernel.py' in hashes,'compiled_kernel_identity_declared')
    if manifest['mode']=='evaluation':
        locked=snapshot/'LOCK.json';protocol=snapshot/'protocol.json'
        audit.check(locked.exists() and protocol.exists(),'frozen_protocol_saved')
        if locked.exists():
            lock=json.loads(locked.read_text())
            for relative,expected in lock['hashes'].items():
                path=snapshot/Path(relative).name
                audit.check(path.exists() and digest(path)==expected,'lock_source_identity',{'file':relative})
        if protocol.exists():
            design=json.loads(protocol.read_text());audit.check(design['cases']==manifest['cases'] and design['methods']==manifest['methods'],'protocol_manifest_identity')
    cases=manifest['cases'];methods=manifest['methods'];rows=[];missing=[];assigned=[]
    for i,case in enumerate(cases):
        subset=P['selected_tail_subset'];tails=i%subset['case_index_modulus']==subset['case_index_remainder']
        for method in methods:
            path=result_dir/f'{case["id"]}__{method}.json';assigned.append(path.name)
            if not path.exists():
                missing.append(path.name)
                if not args.development_smoke:audit.check(False,'missing_assigned_episode',{'file':path.name})
                continue
            row=json.loads(path.read_text());audit.check(row.get('source_hashes')==hashes,'episode_source_identity',{'file':path.name})
            try:rows.append(replay_episode(audit,row,table,case,method,tails))
            except Exception as ex:audit.check(False,'episode_replay_exception',{'file':path.name},repr(ex))
    complete=result_dir/'complete.json'
    if not args.development_smoke:
        audit.check(complete.exists(),'campaign_completion_record')
        if complete.exists():
            done=json.loads(complete.read_text());audit.check(done['episodes']==len(assigned) and done['hashes']==hashes,'completion_identity')
    attempts=[]
    for path in sorted(result_dir.glob('*.interrupted.*.json')):
        saved=json.loads(path.read_text());attempts.append({'file':path.name,'sha256':digest(path),'controller_seconds':saved.get('controller_seconds'),
            'model_calls':saved.get('model_calls'),'completed_stages':len(saved.get('completed_stages',[]))})
    report={'passed':not audit.failures and not audit.qualifications,'hard_checks_passed':not audit.failures,'development_smoke':args.development_smoke,
      'assigned_episodes':len(assigned),'replayed_episodes':len(rows),'missing':missing,'failures':audit.failures,'qualifications':audit.qualifications,'maxima':audit.maxima,
      'episodes':rows,'interrupted_attempt_ledger':attempts,'elapsed_seconds':time.perf_counter()-started,'source_result_dir':str(result_dir),
      'audit_protocol':P,'audit_protocol_sha256':digest(PROTOCOL_PATH),
      'auditor_source_sha256':{p.name:digest(p) for p in (Path(__file__),HERE/'independent_amount_model.py')},
      'runtime':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()},
      'scope':'Numerical replay of supplied inputs. Tail selection is deterministic and prespecified. No plant-setting oracle is supplied to controllers; this retrospective auditor is allowed saved truth.'}
    save(output,report)
    print(json.dumps({k:report[k] for k in ('passed','hard_checks_passed','development_smoke','assigned_episodes','replayed_episodes','missing','failures','qualifications','maxima','elapsed_seconds')},indent=2,default=serial))
if __name__=='__main__':main()
