"""Independent execution of frozen full, open, and ratio-observation policies.

Selection is fixed in protocols/audit_grid_observation*.json before outcomes.
--bind freezes completed inputs, implementation and every assigned policy before
--execute integrates any independent deployment trajectory. Never overwrites an
execution declaration or existing results. Exact cache keys only.
"""
from pathlib import Path
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import warnings
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import engine
import grid_observation as policy_model
import independent_amount_model as independent

DESIGN=ROOT/'protocols/audit_grid_observation.json'
AMENDMENT=ROOT/'protocols/audit_grid_observation_amendment.json'
EXECUTION=ROOT/'audit/grid_observation_execution.json'
RESULTS=ROOT/'audit/grid_observation_deployment.json'
ADAPTIVE=False
KAPPAS=np.array([[1.,1.],[.975,1.025],[1.025,.975]])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def new_json(path,data):
    with path.open('x') as stream:
        json.dump(data,stream,indent=2,allow_nan=False,default=serial);stream.write('\n')


def save_checkpoint(data):
    temporary=RESULTS.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False,default=serial)+'\n')
    temporary.replace(RESULTS)


def expected_sources():
    d=json.loads(DESIGN.read_text()); sources=d['source_sha256'].copy()
    if AMENDMENT.exists():sources.update(json.loads(AMENDMENT.read_text())['source_sha256_overrides'])
    return d,sources


def bind():
    if EXECUTION.exists() or RESULTS.exists():raise FileExistsError('Execution declaration or results already exist')
    design,sources=expected_sources()
    for name,expected in sources.items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Frozen source differs: '+name)
    source=ROOT/('results/grid_observation_adaptive.json' if ADAPTIVE else 'results/grid_observation.json')
    data=json.loads(source.read_text())
    if not ADAPTIVE and not data.get('complete'):raise RuntimeError('Main grid study incomplete')
    gate=json.loads((ROOT/'audit/grid_observation_integrator_gate.json').read_text())
    if not gate.get('passed'):raise RuntimeError('Preflight gate did not pass')
    binding={name:sha(ROOT/name) for name in sources}
    for p in [DESIGN,Path(__file__),source]:binding[str(p.relative_to(ROOT))]=sha(p)
    if AMENDMENT.exists():binding[str(AMENDMENT.relative_to(ROOT))]=sha(AMENDMENT)
    roots={r['id']:r for r in policy_model.roots()}
    records=([r for part in data['results'] if part['complete'] for r in part['records']]
             if ADAPTIVE else data['records'])
    assigned=[]; coverage=[]; trees={}
    primary_keys={(r['root'],r['grid'],r['mode'],r['epsilon']) for r in records if r['grid'] in [41,81,161]}
    if not ADAPTIVE and len(primary_keys)!=126:raise ValueError('Expected126 primary root/grid/information records')
    if ADAPTIVE and any(r['grid']!=321 for r in records):raise ValueError('Adaptive record grid is not321')

    def saved_subtree(root,grid):
        key=(root,grid)
        if key in trees:return trees[key]
        available=[g for g in [grid,161,321] if g>=grid and (ROOT/'results/trees'/f'{root}_{g}.npz').exists()]
        if not available:raise FileNotFoundError('No complete tree for '+str(key))
        path=ROOT/'results/trees'/f'{root}_{min(available)}.npz';metadata=path.with_suffix('.json')
        meta=json.loads(metadata.read_text())
        if meta.get('complete') is False or meta['npz_sha256']!=sha(path):raise RuntimeError('Incomplete or changed tree')
        for p in [path,metadata]:binding[str(p.relative_to(ROOT))]=sha(p)
        with np.load(path) as z:tree={k:z[k] for k in z.files}
        trees[key]=policy_model.subset_tree(tree,grid)
        return trees[key]

    def add(record,cohort,policy,requested_log_target,selection):
        assigned.append({'id':len(assigned),'root':record['root'],'family':record['family'],
                         'grid':record['grid'],'mode':record['mode'],'epsilon':record['epsilon'],
                         'cohort':cohort,'target':policy['target'],'requested_log_target':requested_log_target,
                         'actual_log_target':math.log(policy['target']),
                         'target_round_trip_shift':math.log(policy['target'])-requested_log_target,
                         'selection':selection,'policy':policy})

    for record in records:
        grid=record['grid']
        if grid not in [41,81,161,321]:raise ValueError('Unexpected grid')
        cohort='fixed' if grid!=321 else 'adaptive_fixed'
        if len(record['fixed_policies'])!=6:raise ValueError('Incomplete fixed-target record')
        for p in record['fixed_policies']:
            add(record,cohort,p,math.log(p['target']),['predeclared_fixed_target'])
        if grid not in [161,321]:continue
        intervals=record['intervals']; selected={}
        indices=sorted({0,(len(intervals)-1)//2,len(intervals)-1}) if intervals else []
        for index in indices:
            lo,hi=intervals[index]
            for label,z in [('lower',lo),('midpoint',lo+(hi-lo)/2),('upper',hi)]:
                selected.setdefault(float(z),[]).append({'component':index,'position':label})
        coverage.append({'root':record['root'],'grid':grid,'mode':record['mode'],'epsilon':record['epsilon'],
                         'total_components':len(intervals),'selected_component_indices':indices,
                         'distinct_target_points':len(selected),'empty_set_retained':not bool(intervals)})
        if selected:
            subtree=saved_subtree(record['root'],grid)
            for z,selection in sorted(selected.items()):
                p=policy_model.fixed_policy(subtree,math.exp(z),record['mode'],record['epsilon'])
                add(record,'component' if grid!=321 else 'adaptive_component',p,z,selection)
    if not ADAPTIVE and sum(c['cohort']=='fixed' for c in assigned)!=756:raise ValueError('Fixed denominator is not756')
    if ADAPTIVE and sum(c['cohort']=='adaptive_fixed' for c in assigned)!=42*sum(p['complete'] for p in data['results']):
        raise ValueError('Adaptive fixed denominator mismatch')
    for name,expected in binding.items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Source changed while binding: '+name)
    execution={'declared_utc':datetime.now(timezone.utc).isoformat(),
               'status':'Complete case list and implementation frozen before independent deployment',
               'input_sha256':binding,'design':design,'root_states':{name:r['state'] for name,r in roots.items()},
               'assigned_cases':assigned,'component_coverage':coverage,
               'counts':dict(collections.Counter(c['cohort'] for c in assigned)),
               'main_adaptive_status':({'triggered_roots':data['triggered_roots'],'complete':data['complete'],
                                       'unresolved':[p for p in data['results'] if not p['complete']]}
                                      if ADAPTIVE else None)}
    new_json(EXECUTION,execution)
    print(json.dumps({'bound':str(EXECUTION),'counts':execution['counts']},indent=2))


def compare(left,right,t):
    a,b=np.asarray(left),np.asarray(right); error=np.abs(a-b)
    allowed=t['amount_absolute_g']+t['amount_relative']*np.abs(b[1:])
    zerror=abs(math.log(a[1]/a[2])-math.log(b[1]/b[2]))
    return {'passed':bool(np.all(error[1:]<=allowed) and error[0]<=t['volume_absolute_L'] and zerror<=t['log_ratio_absolute']),
            'component_errors':error.tolist(),'amount_allowances':allowed.tolist(),
            'max_amount_error_over_allowance':float(np.max(error[1:]/allowed)),
            'log_ratio_error':float(zerror)}


def physical(state,minimum,t):
    return bool(np.all(np.isfinite(state)) and min(state[:3])>0 and state[0]<=t['physical_volume_max_L']
                and minimum>=t['physical_substrate_min_g_L'])


def noise_cells(actual_z,epsilon,policy):
    lo=actual_z-epsilon; hi=actual_z+epsilon
    z=np.asarray(policy['predicted_z']);radius=policy['effective_radius']
    breaks=np.r_[z-radius,z+radius]
    cuts=sorted(set([float(lo),float(hi)]+[float(x) for x in breaks if lo<=x<=hi]))
    points=[{'kind':'endpoint','lower':x,'upper':x,'measured_z':x} for x in cuts]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if a<b:points.append({'kind':'open_cell','lower':a,'upper':b,'measured_z':a+(b-a)/2})
    return sorted(points,key=lambda p:(p['measured_z'],p['kind']))


def execute():
    if RESULTS.exists():raise FileExistsError('Audit result already exists')
    execution=json.loads(EXECUTION.read_text());execution_hash=sha(EXECUTION)
    for name,expected in execution['input_sha256'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Frozen execution input differs: '+name)
    design=execution['design'];t=design['thresholds'];ref=design['transition_reference'];eng=design['matching_engine']
    independent.RTOL=ref['requested_rtol'];independent.ATOL=ref['atol'];independent.MAX_STEP=ref['maxstep_h']
    report={'complete':False,'passed':False,'execution_sha256':execution_hash,
            'input_sha256_before':execution['input_sha256'],'cases':[],'transitions':[],
            'controller_maps':[],'failures':[],'band_label_differences':[],
            'runtime':{'python':sys.version,'executable':sys.executable,**{n:importlib.metadata.version(n) for n in ['numpy','scipy','numba']}},
            'scope':design['scope']}
    new_json(RESULTS,report)
    models={f:independent.AmountReactor(f) for f in ['analytic','table']}
    arguments={f:engine.arguments(f) for f in ['analytic','table']}
    transition_cache={};map_cache={};counts=collections.Counter();seconds=collections.Counter()
    started=time.perf_counter()

    def fail(code,**context):report['failures'].append({'code':code,**context})

    def transition(state,feed,kappa,family):
        counts['logical_transition_requests']+=1
        key=(family,np.asarray(state,dtype=float).tobytes(),float(feed),tuple(kappa))
        if key in transition_cache:
            counts['transition_cache_hits']+=1
            return report['transitions'][transition_cache[key]]
        ident=len(report['transitions']);transition_cache[key]=ident
        row={'id':ident,'family':family,'input_amount_state':np.asarray(state).tolist(),
             'feed':float(feed),'kappa':list(kappa),'complete':False}
        report['transitions'].append(row)
        tick=time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter('always');out=models[family].interval(state,feed,kappa)
            row['independent']={'amount_state':out['amount_state'].tolist(),'min_S':float(out['min_S']),
                                'max_S':float(out['max_S']),'rhs_calls':int(out['rhs_calls']),
                                'physical_valid':physical(out['amount_state'],out['min_S'],t),
                                'warnings':[str(w.message) for w in ww]}
            if not row['independent']['physical_valid']:fail('independent_physical_invalid',transition_id=ident)
        except Exception as error:
            row['independent_exception']={'type':type(error).__name__,'message':str(error)}
            fail('independent_exception',transition_id=ident,**row['independent_exception'])
        seconds['independent_intervals']+=time.perf_counter()-tick
        tick=time.perf_counter()
        try:
            endpoint,minimum,accepted,rejected=engine.flow(independent.to_log(state),feed,np.asarray(kappa),arguments[family],eng['rtol'],eng['atol'],eng['maxstep_h'])
            amounts=independent.from_log(endpoint)
            row['engine']={'amount_state':amounts.tolist(),'min_S':float(minimum),
                           'accepted_steps':int(accepted),'rejected_steps':int(rejected),
                           'physical_valid':physical(amounts,minimum,t)}
            if not row['engine']['physical_valid']:fail('engine_physical_invalid',transition_id=ident)
            if 'independent' in row:
                row['comparison']=compare(amounts,row['independent']['amount_state'],t)
                if not row['comparison']['passed']:fail('matched_numerical_discrepancy',transition_id=ident,comparison=row['comparison'])
        except Exception as error:
            row['engine_exception']={'type':type(error).__name__,'message':str(error)}
            fail('engine_exception',transition_id=ident,**row['engine_exception'])
        seconds['matched_engine_intervals']+=time.perf_counter()-tick
        row['complete']='independent' in row and 'engine' in row
        return row

    def full_state_action(state,family,grid,target):
        # API receives actual observed state but never the true disturbance ID.
        counts['full_state_decision_requests']+=1
        key=(family,int(grid),np.asarray(state,dtype=float).tobytes())
        if key in map_cache:
            mapped=map_cache[key];counts['controller_map_cache_hits']+=1
        else:
            tick=time.perf_counter();logstate=independent.to_log(state)
            actions=policy_model.alphabet(logstate,grid)
            out,minimum,accepted,rejected=engine.batch(np.repeat(logstate[None,:],len(actions)*3,axis=0),
                np.repeat(actions,3),np.tile(KAPPAS,(len(actions),1)),arguments[family],eng['rtol'],eng['atol'],eng['maxstep_h'])
            out=out.reshape(len(actions),3,4);minimum=minimum.reshape(len(actions),3)
            valid=np.all(np.isfinite(out),axis=-1)&(out[...,0]>0)&(out[...,0]<=t['physical_volume_max_L'])&(minimum>=t['physical_substrate_min_g_L'])
            mapped={'id':len(report['controller_maps']),'actions':actions,'z':out[...,1]-out[...,2],
                    'valid':np.all(valid,axis=1)}
            report['controller_maps'].append({'id':mapped['id'],'family':family,'grid':grid,'observed_amount_state':np.asarray(state).tolist(),
                                             'actions':actions.tolist(),'transitions':len(actions)*3,
                                             'accepted_steps':int(accepted),'rejected_steps':int(rejected)})
            map_cache[key]=mapped;seconds['full_state_maps']+=time.perf_counter()-tick
        error=np.max(np.abs(mapped['z']-math.log(target)),axis=1);error[~mapped['valid']]=np.inf
        index=int(np.argmin(error))
        if not np.isfinite(error[index]):raise ValueError('Full-state minimax has no physical action')
        return float(mapped['actions'][index]),{'controller_map':mapped['id'],'action_index':index,'predicted_one_step_error':float(error[index])}

    for assigned in execution['assigned_cases']:
        case={k:v for k,v in assigned.items() if k!='policy'};p=assigned['policy'];case['predicted_policy']=p
        case.update(complete=False,first_stages=[],episodes=[],exceptions=[])
        report['cases'].append(case);complete=True
        if p.get('unresolved'):
            fail('unresolved_assigned_policy',case_id=case['id']);case['exceptions'].append({'message':'Frozen policy unresolved'})
            continue
        root_state=independent.from_log(execution['root_states'][case['root']]);zstar=math.log(case['target'])
        for j,k0 in enumerate(KAPPAS):
            first=transition(root_state,p['first_feed'],k0,case['family'])
            stage={'affinity_index':j,'transition_id':first['id'],'measurement_decisions':[]};case['first_stages'].append(stage)
            if 'independent' not in first:
                complete=False;continue
            state=np.array(first['independent']['amount_state']);z=math.log(state[1]/state[2])
            points=noise_cells(z,case['epsilon'],p) if case['mode']=='ratio' else [{'kind':case['mode'],'measured_z':None}]
            for observation in points:
                counts['observation_decisions']+=1
                decision=dict(observation);decision['id']=len(stage['measurement_decisions']);stage['measurement_decisions'].append(decision)
                try:
                    if case['mode']=='ratio':
                        feed=policy_model.choose_second(p,observation['measured_z'])
                        centers=np.asarray(p['predicted_z']);radius=p['effective_radius']
                        decision['compatible_mask']=np.flatnonzero((centers-radius<=observation['measured_z'])&(observation['measured_z']<=centers+radius)).tolist()
                        decision['actual_observation_error']=observation['measured_z']-z
                    elif case['mode']=='open':
                        if len(p['second_actions'])!=1:raise ValueError('Open plan does not have exactly one second action')
                        feed=p['second_actions'][0]['feed']
                    else:
                        feed,detail=full_state_action(state,case['family'],case['grid'],case['target']);decision.update(detail)
                    if feed<0 or feed>.2 or state[0]+2*feed>t['physical_volume_max_L']:
                        raise ValueError('Controller feed violates physical cap')
                    decision['feed']=float(feed)
                except Exception as error:
                    complete=False;decision['exception']={'type':type(error).__name__,'message':str(error)}
                    case['exceptions'].append({'first_affinity_index':j,'decision':decision['id'],**decision['exception']})
                    fail('controller_exception',case_id=case['id'],first_affinity_index=j,**decision['exception'])
                    continue
                for k,k1 in enumerate(KAPPAS):
                    out=transition(state,feed,k1,case['family'])
                    episode={'first_affinity_index':j,'measurement_decision':decision['id'],'second_affinity_index':k,
                             'transition_id':out['id'],'complete':'independent' in out}
                    case['episodes'].append(episode)
                    if 'independent' not in out:complete=False;continue
                    end=np.array(out['independent']['amount_state']);error=abs(math.log(end[1]/end[2])-zstar)
                    valid=first['independent']['physical_valid'] and out['independent']['physical_valid']
                    episode.update(error=float(error),biomass_gDW=float(end[1]+end[2]),physical_valid=valid,
                                   bands={b:bool(valid and error<=t[b]) for b in ['inner','outer','actual']})
        case['complete']=complete and len(case['first_stages'])==3 and bool(case['episodes']) and all(e['complete'] for e in case['episodes'])
        done=[e for e in case['episodes'] if e['complete']]
        case['maximum_error']=max((e['error'] for e in done),default=None)
        case['minimum_biomass_gDW']=min((e['biomass_gDW'] for e in done),default=None)
        case['all_paths_bands']={b:bool(case['complete'] and all(e['bands'][b] for e in done)) for b in ['inner','outer','actual']}
        for band in ['inner','outer','actual']:
            predicted=bool(p[band]);actual=case['all_paths_bands'][band]
            if predicted!=actual:
                report['band_label_differences'].append({'case_id':case['id'],'cohort':case['cohort'],'band':band,'predicted':predicted,'actual':actual,
                                                       'predicted_error':p['worst_error'],'actual_error':case['maximum_error']})
            if predicted and not actual:fail('predicted_feasible_deployment_failure',case_id=case['id'],band=band,cohort=case['cohort'],predicted_error=p['worst_error'],actual_error=case['maximum_error'])
        if (case['id']+1)%20==0:
            report['progress']={'completed_cases':len(report['cases']),'assigned_cases':len(execution['assigned_cases']),
                                'unique_transitions':len(report['transitions']),'seconds':time.perf_counter()-started}
            save_checkpoint(report);print(json.dumps(report['progress']),flush=True)
    comparisons=[r['comparison'] for r in report['transitions'] if 'comparison' in r]
    groups={}
    for case in report['cases']:
        key=(case['cohort'],case['grid'],case['mode'],case['epsilon'])
        groups.setdefault(key,[]).append(case)
    report['cohort_summary']=[{'cohort':key[0],'grid':key[1],'mode':key[2],'epsilon':key[3],
        'problems':len(cases),'complete':sum(c['complete'] for c in cases),
        'all_paths_success':{b:sum(c.get('all_paths_bands',{}).get(b,False) for c in cases) for b in ['inner','outer','actual']},
        'predicted_all_paths_success':{b:sum(c['predicted_policy'].get(b,False) for c in cases) for b in ['inner','outer','actual']}}
        for key,cases in groups.items()]
    report['numerical_summary']={'unique_transitions':len(report['transitions']),'complete':sum(r['complete'] for r in report['transitions']),
        'comparisons':len(comparisons),'comparisons_passed':sum(c['passed'] for c in comparisons),
        'maximum_amount_error_over_allowance':max((c['max_amount_error_over_allowance'] for c in comparisons),default=None),
        'maximum_log_ratio_error':max((c['log_ratio_error'] for c in comparisons),default=None)}
    report['counts']=dict(counts);report['timing_seconds']=dict(seconds)
    report['input_sha256_after']={name:sha(ROOT/name) for name in execution['input_sha256']}
    report['inputs_unchanged']=report['input_sha256_after']==report['input_sha256_before'] and sha(EXECUTION)==execution_hash
    report['complete']=len(report['cases'])==len(execution['assigned_cases']) and all(c['complete'] for c in report['cases'])
    report['numerical_checks_passed']=len(comparisons)==len(report['transitions']) and all(c['passed'] for c in comparisons)
    report['passed']=report['complete'] and report['inputs_unchanged'] and not report['failures']
    report['elapsed_seconds']=time.perf_counter()-started
    save_checkpoint(report)
    print(json.dumps({k:report[k] for k in ['complete','passed','inputs_unchanged','numerical_summary','elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bind',action='store_true');group.add_argument('--execute',action='store_true')
    parser.add_argument('--adaptive',action='store_true')
    option=parser.parse_args()
    if option.adaptive:
        ADAPTIVE=True
        EXECUTION=ROOT/'audit/grid_observation_adaptive_execution.json'
        RESULTS=ROOT/'audit/grid_observation_adaptive_deployment.json'
    bind() if option.bind else execute()
