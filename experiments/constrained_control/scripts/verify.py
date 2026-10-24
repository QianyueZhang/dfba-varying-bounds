#!/usr/bin/env python3
"""Audit the released constrained-control data and optionally integrate its policies."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
from simulator import Reactor, features, initial_from_condition
from surrogate_model import load_models

PACKAGE = Path(__file__).resolve().parents[1]
REFERENCE = PACKAGE/'reference'
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def audit(integrate=False):
    started = time.perf_counter()
    checks = []
    def add(name, passed, details=None):
        checks.append({'name': name, 'passed': bool(passed), 'details': details})
    p = load(REFERENCE/'protocol.json')
    methods = load(REFERENCE/'protocol_amendment_02.json')['changes']['methods']
    add('response table hash', sha(PACKAGE/'data/flux_table_eps1e-06.json') == p['source_table_sha256'])
    manifest = load(PACKAGE/'SOURCE_MANIFEST.json')
    differences = [r['release'] for r in manifest['files'] if not (PACKAGE/r['release']).is_file() or sha(PACKAGE/r['release']) != r['release_sha256']]
    add('distributed source/data hashes', not differences, differences)
    arrays = {name: np.load(REFERENCE/'data'/f'{name}.npz', allow_pickle=False) for name in ['training','validation','test']}
    for name, count, groups in [('training',1152,24),('validation',192,6),('test',256,4)]:
        a = arrays[name]
        add(name+' grouped allocation', a['X'].shape == (count,9) and a['Y'].shape == (count,2) and len(set(a['condition_index'])) == groups)
        add(name+' feed bounds and volume cap', np.all(a['controls'] >= 0) and np.all(a['controls'] <= .2) and np.all(a['controls'].sum(axis=1) <= .65+1e-12))
        expected = np.asarray([features(u,p['conditions'][name][int(ci)]) for u,ci in zip(a['controls'],a['condition_index'])])
        add(name+' features match controls/conditions', np.array_equal(expected,a['X']))
    sets = [set(map(tuple,p['conditions'][name])) for name in arrays]
    add('condition groups disjoint', not any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)]))
    selection = load(REFERENCE/'results/selection_lock.json')['selection']
    models = load_models(REFERENCE/'results/locked_models.json')
    for name, model in models.items():
        r = model.record
        add(name+' training-only standardisation', np.allclose(r['input_mean'], arrays['training']['X'].mean(axis=0),rtol=0,atol=1e-12) and np.allclose(r['output_mean'],arrays['training']['Y'].mean(axis=0),rtol=0,atol=1e-12))
        expected = np.load(REFERENCE/'data'/f'{name}_test_predictions.npz',allow_pickle=False)
        pred = model.predict(arrays['test']['X'])
        error = float(np.max(abs(pred-expected['prediction'])))
        add(name+' JSON predictor agreement', error < 1e-12, {'maximum_absolute_error':error})
        add(name+' validation-based selection', selection[name]['selected']['validation_joint_standardized_mse'] == min(c['validation_joint_standardized_mse'] for c in selection[name]['candidates']))
        metrics = load(REFERENCE/'results/prediction_metrics.json')[name]
        rmse = np.sqrt(np.mean((pred-arrays['test']['Y'])**2,axis=0))
        add(name+' prediction metrics', abs(rmse[0]-metrics['log_ratio_rmse']) < 1e-12 and abs(rmse[1]-metrics['log_biomass_amount_rmse']) < 1e-12)
    paths = sorted((REFERENCE/'results/policies').glob('*.json'))
    rows = [load(path) for path in paths]
    actual = {(r['condition_id'],r['target_ratio'],r['seed'],r['method']) for r in rows}
    expected = {(c,t,s,m) for c in range(4) for t in [.5,1.,2.] for s in [3101,3102,3103] for m in methods}
    add('complete unique policy coverage', len(rows) == 180 and actual == expected)
    bad = []
    for path, r in zip(paths,rows):
        q = r['tight_verification']; v,x,y,s = q['terminal']; z = math.log(x/y)
        legal = all(0 <= u <= .2 for u in r['controls']) and sum(r['controls']) <= .65+5e-9
        ok = legal and q['status']=='completed' and v > 0 and min(x,y) > 0 and s >= -1e-6
        ok &= np.array_equal(r['initial'],initial_from_condition(r['condition']))
        ok &= abs(v-(.5+2*sum(r['controls']))) < 1e-10 and abs(v*(x+y)-q['biomass_amount_gDW']) < 1e-10
        ok &= abs(z-q['z']) < 1e-12
        feasible = abs(z-math.log(r['target_ratio'])) <= math.log(1.01) and v <= 1.8+1e-8
        ok &= feasible == q['feasible'] and r['total_mechanistic_attempts'] <= 300
        ok &= r['total_mechanistic_attempts'] == len(r['trace'])+len(r['failed_evaluations'])
        for trial in r['trace']:
            tv,tx,ty,ts = trial['terminal']
            ok &= abs(tv-(.5+2*sum(trial['control']))) < 1e-10 and abs(tv*(tx+ty)-trial['biomass_amount_gDW']) < 1e-10 and abs(math.log(tx/ty)-trial['z']) < 1e-12
        if not ok: bad.append(path.name)
    add('all policy bounds/balances/ratios/budgets',not bad,bad)
    summary = load(REFERENCE/'results/summary.json')
    for name in methods:
        rows_m = [r for r in rows if r['method']==name]; s = next(x for x in summary['method_summary'] if x['method']==name)
        total = sum(r['online_total_seconds'] for r in rows_m)
        add(name+' summary feasibility/cost', len(rows_m)==s['runs']==36 and sum(r['tight_verification']['feasible'] for r in rows_m)==s['verified_feasible'] and abs(total-s['sum_online_seconds']) < 1e-8)
    sens = load(REFERENCE/'results/initial_condition_sensitivity.json')['cases']
    add('complete fixed-policy sensitivity coverage', len(sens)==240 and len({(r['case_id'],r['method'],r['ratio_multiplier'],r['glucose_multiplier']) for r in sens})==240)
    for name in ['direct_multistart','mlp']:
        d = load(REFERENCE/'results'/f'direct_{name}.json'); inp = load(REFERENCE/'results'/f'direct_input_{name}.json')
        r = next(r for r in rows if r['condition_id']==0 and r['target_ratio']==1 and r['seed']==3101 and r['method']==name)
        add(name+' direct prespecified identity',d['controls_L_h']==inp['controls']==r['controls'] and d['initial_V_XSC_XKP_S']==inp['initial']==r['initial'])
        v,x,y,s=d['terminal']
        add(name+' direct endpoint feasibility',abs(v*(x+y)-d['terminal_biomass_gDW'])<1e-10 and abs(math.log(x/y)-math.log(d['target_ratio'])) <= math.log(1.01) and v <= 1.8+1e-8 and min(x,y,s)>0)
        add(name+' direct trajectory hash',sha(REFERENCE/'results'/f'direct_{name}_trajectory.csv')==d['trajectory_sha256'])
    integrations = []
    if integrate:
        reactor = Reactor()
        for path,r in zip(paths,rows):
            q=reactor.simulate(r['controls'],r['initial'],rtol=1e-9,atol=1e-11)
            saved=r['tight_verification']
            err=float(np.max(abs(np.asarray(q['terminal'])-saved['terminal'])))
            feasible=abs(q['z']-math.log(r['target_ratio'])) <= math.log(1.01) and q['terminal'][0] <= 1.8+1e-8
            integrations.append({'policy':path.name,'maximum_terminal_absolute_difference':err,'feasibility_agrees':feasible==saved['feasible']})
        add('fresh integration of all 180 policies',len(integrations)==180 and all(r['feasibility_agrees'] and r['maximum_terminal_absolute_difference']<1e-8 for r in integrations),{'maximum_terminal_absolute_difference':max(r['maximum_terminal_absolute_difference'] for r in integrations)})
    return {'passed':all(c['passed'] for c in checks),'checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),'scope':'Supplied study data, fitted JSON predictors, policy arithmetic, summaries and direct endpoint records; optional fresh table-ODE integration. No new optimisation or direct KKT trajectory run.','integrated_policies':len(integrations),'elapsed_seconds':time.perf_counter()-started,'checks':checks,'integrations':integrations}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--integrate',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output and args.output.resolve().is_relative_to(REFERENCE):
        p.error('Write verification reports outside the supplied reference records.')
    report=audit(args.integrate)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checks','integrations']},indent=2))
    if not report['passed']:
        print(json.dumps([c for c in report['checks'] if not c['passed']],indent=2));raise SystemExit(1)


if __name__=='__main__':main()
