"""Frozen numerical preflight before new controller outcome evaluation.

Use --declare followed by --execute. Outputs are never overwritten. Uniform
roots are prescribed, not regenerated or adjusted using numerical outcomes.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import engine
import independent_amount_model as independent

PROTOCOL = ROOT / 'protocols/audit_numerical_preflight.json'
RESULTS = ROOT / 'audit/grid_observation_integrator_gate.json'
SOURCES = ['scripts/engine.py', 'audit/independent_amount_model.py',
           'vendor/prior_kernel.py', 'data/flux_table_eps1e-06.json',
           'data/prescribed_roots.json', 'audit/grid_observation_preflight.py']
ENGINE = {'rtol': 2e-13, 'atol': 2e-15, 'maxstep_h': .01}
REFERENCE = {'requested_rtol': 2e-14, 'effective_rtol': float(100*np.finfo(float).eps),
             'atol': 2e-15, 'maxstep_h': .005}
THRESHOLDS = {'amount_absolute_g': 2e-7, 'amount_relative': 2e-8,
              'log_ratio_absolute': 1e-7, 'volume_absolute_L': 2e-13,
              'physical_substrate_min_g_L': -2e-7, 'volume_cap_L': 1.8}
KAPPAS = [[1., 1.], [.975, 1.025], [1.025, .975]]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def roots():
    result = [r for r in json.loads((ROOT/'data/prescribed_roots.json').read_text()) if r['timing']=='uniform']
    assert len(result)==6 and len({r['id'] for r in result})==6
    assert {(r['family'],r['volume']) for r in result} == {(f,v) for f in ['analytic','table'] for v in [.9,1.3,1.7]}
    return result


def cap(state):
    return max(0., min(.2, (1.8-float(state[0])-1e-11)/2.))


def declare():
    if PROTOCOL.exists() or RESULTS.exists():
        raise FileExistsError('Preflight protocol or results already exist')
    rs = roots()
    protocol = {
        'declared_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'Frozen before preflight integrations and controller outcome evaluation',
        'source_sha256': {p:sha(ROOT/p) for p in SOURCES},
        'root_ids': [r['id'] for r in rs],
        'roots_are_prescribed': True,
        'engine': ENGINE, 'independent_DOP853': REFERENCE, 'thresholds': THRESHOLDS,
        'affinities': KAPPAS, 'feed_fractions_of_current_cap': [0.,.5,1.],
        'primary_rule': 'All six prescribed uniform-history roots, all three feed fractions of clipped capacity, and all three affinities: 54 matched intervals.',
        'second_state_rule': 'For each root take (a) the lowest-substrate endpoint among the three nominal-affinity primary feeds, ties by feed fraction index, and (b) the nominal-affinity full-capacity-feed endpoint. Use the independent endpoint as the COMMON input to both solvers. At each selected second state test all three current-cap feed fractions and all three affinities:108 additional logical comparisons. Retain both labels if states coincide; do not count them as independent biological replicates.',
        'expected_logical_comparisons':162,
        'comparison_rule': 'Absolute amount error <=2e-7+2e-8*abs(reference amount), log(M1/M2) error<=1e-7, volume error<=2e-13. All endpoints and sampled substrate minima must meet physical checks. Feed cap max(0,min(.2,(1.8-V-1e-11)/2)).',
        'gate_rule': 'Pass only if all162 comparisons complete and pass, inputs and protocol remain unchanged, and no physical or numerical failure occurs. Preserve every failure. A pass establishes sampled consistency only, not a continuous-domain certificate.',
        'failure_rule': 'Do not retune models, roots, tolerances, case selection or thresholds after seeing outcomes. Any additional diagnosis gets its own frozen record and does not overwrite this result.',
    }
    new_json(PROTOCOL, protocol)
    print(json.dumps({'declared':str(PROTOCOL),'roots':protocol['root_ids'],'comparisons':162}))


def compare(left,right):
    a,b=np.asarray(left),np.asarray(right)
    errors=np.abs(a-b)
    allowance=THRESHOLDS['amount_absolute_g']+THRESHOLDS['amount_relative']*np.abs(b[1:])
    zerror=abs(math.log(a[1]/a[2])-math.log(b[1]/b[2]))
    passed=bool(np.all(errors[1:]<=allowance) and errors[0]<=THRESHOLDS['volume_absolute_L'] and zerror<=THRESHOLDS['log_ratio_absolute'])
    return {'passed':passed,'component_errors':errors.tolist(),'amount_allowances':allowance.tolist(),
            'max_amount_error_over_allowance':float(np.max(errors[1:]/allowance)),
            'log_ratio_error':float(zerror)}


def physical(state,minimum):
    return bool(np.all(np.isfinite(state)) and min(state[:3])>0 and state[0]<=1.8
                and minimum>=THRESHOLDS['physical_substrate_min_g_L'])


def execute():
    if RESULTS.exists():raise FileExistsError('Preflight results already exist')
    p=json.loads(PROTOCOL.read_text()); ph=sha(PROTOCOL)
    for name,expected in p['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Frozen input changed: '+name)
    if p['engine']!=ENGINE or p['independent_DOP853']!=REFERENCE or p['thresholds']!=THRESHOLDS:
        raise RuntimeError('Configuration differs from frozen protocol')
    independent.RTOL=REFERENCE['requested_rtol']; independent.ATOL=REFERENCE['atol']; independent.MAX_STEP=REFERENCE['maxstep_h']
    report={'complete':False,'passed':False,'protocol_sha256':ph,'source_sha256':p['source_sha256'],'source_sha256_before':p['source_sha256'],
            'records':[],'second_state_selection':[],'failures':[],
            'runtime':{'python':sys.version,'executable':sys.executable,**{x:importlib.metadata.version(x) for x in ['numpy','scipy','numba']}}}
    started=time.perf_counter(); cache={}; unique=0

    def evaluate(state,feed,kappa,family,context):
        nonlocal unique
        key=(family,tuple(float(v) for v in state),float(feed),tuple(kappa))
        if key in cache:
            cached=cache[key]
            row={'id':len(report['records']),'context':context,'reused_unique_transition':cached['unique_transition'],**{k:v for k,v in cached.items() if k not in ['id','context','reused_unique_transition']}}
            report['records'].append(row)
            return row
        row={'id':len(report['records']),'unique_transition':unique,'context':context,
             'family':family,'input_amount_state':np.asarray(state).tolist(),'feed':float(feed),'kappa':kappa,'complete':False}
        unique+=1; report['records'].append(row); cache[key]=row
        try:
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter('always')
                ref=independent.AmountReactor(family).interval(state,feed,np.asarray(kappa))
            row['reference']={'state':ref['amount_state'].tolist(),'min_S':float(ref['min_S']),'max_S':float(ref['max_S']),
                              'rhs_calls':int(ref['rhs_calls']),'seconds':time.perf_counter()-tick,
                              'warnings':[str(w.message) for w in ww], 'physical_valid':physical(ref['amount_state'],ref['min_S'])}
            tick=time.perf_counter()
            end,minimum,steps,rejected=engine.flow(independent.to_log(state),feed,np.asarray(kappa),engine.arguments(family),ENGINE['rtol'],ENGINE['atol'],ENGINE['maxstep_h'])
            amounts=independent.from_log(end)
            row['engine']={'state':amounts.tolist(),'min_S':float(minimum),'accepted_steps':int(steps),'rejected_steps':int(rejected),
                           'seconds':time.perf_counter()-tick,'physical_valid':physical(amounts,minimum)}
            row['comparison']=compare(amounts,ref['amount_state']); row['complete']=True
            if not row['comparison']['passed'] or not row['reference']['physical_valid'] or not row['engine']['physical_valid']:
                report['failures'].append({'record':row['id'],'unique_transition':row['unique_transition'],'context':context,'kind':'numerical_or_physical_failure'})
        except Exception as error:
            row['exception']={'type':type(error).__name__,'message':str(error)}
            report['failures'].append({'record':row['id'],'context':context,'kind':'exception',**row['exception']})
        return row

    for root in roots():
        state=independent.from_log(root['state']); maximum=cap(state); nominal=[]
        for i,fraction in enumerate([0.,.5,1.]):
            for j,kappa in enumerate(KAPPAS):
                row=evaluate(state,maximum*fraction,kappa,root['family'],{'root':root['id'],'cohort':'prescribed','feed_fraction_index':i,'affinity_index':j})
                if j==0:nominal.append(row)
        if any(not x['complete'] for x in nominal):continue
        low=min(range(3),key=lambda i:(nominal[i]['reference']['state'][3]/nominal[i]['reference']['state'][0],i))
        for label,selected in [('lowest_substrate_nominal_successor',low),('maximum_feed_nominal_successor',2)]:
            source=nominal[selected]; second=np.asarray(source['reference']['state']); maximum=cap(second)
            report['second_state_selection'].append({'root':root['id'],'label':label,'source_record':source['id'],
                                                      'amount_state':second.tolist(),'substrate_g_L':float(second[3]/second[0]),'feed_cap':maximum})
            for i,fraction in enumerate([0.,.5,1.]):
                for j,kappa in enumerate(KAPPAS):
                    evaluate(second,maximum*fraction,kappa,root['family'],{'root':root['id'],'cohort':label,'feed_fraction_index':i,'affinity_index':j})
    comparisons=[r['comparison'] for r in report['records'] if 'comparison' in r]
    report['summary']={'assigned_logical_comparisons':162,'completed':sum(r['complete'] for r in report['records']),
                       'comparisons_passed':sum(c['passed'] for c in comparisons),'unique_transitions':unique,
                       'maximum_amount_error_over_allowance':max((c['max_amount_error_over_allowance'] for c in comparisons),default=None),
                       'maximum_log_ratio_error':max((c['log_ratio_error'] for c in comparisons),default=None),
                       'maximum_component_error':np.max([c['component_errors'] for c in comparisons],axis=0).tolist() if comparisons else None,
                       'minimum_selected_second_substrate':min((s['substrate_g_L'] for s in report['second_state_selection']),default=None),
                       'maximum_selected_second_volume':max((s['amount_state'][0] for s in report['second_state_selection']),default=None)}
    report['source_sha256_after']={p:sha(ROOT/p) for p in SOURCES}
    report['inputs_unchanged']=report['source_sha256_after']==report['source_sha256_before'] and sha(PROTOCOL)==ph
    report['complete']=len(report['records'])==162 and all(r['complete'] for r in report['records'])
    report['passed']=report['complete'] and report['inputs_unchanged'] and not report['failures']
    report['elapsed_seconds']=time.perf_counter()-started
    new_json(RESULTS,report)
    print(json.dumps({k:report[k] for k in ['complete','passed','inputs_unchanged','summary','failures','elapsed_seconds']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--declare',action='store_true');group.add_argument('--execute',action='store_true')
    options=parser.parse_args()
    declare() if options.declare else execute()
