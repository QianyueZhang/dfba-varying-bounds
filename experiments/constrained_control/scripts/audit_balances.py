#!/usr/bin/env python3
"""Read-only scientific checks and independent amount-balance integration.
The completed optimisation campaign is never modified or rerun by this script.
"""
from pathlib import Path
import json,sys,hashlib,time,os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE/'reference'
HERE=Path(os.environ.get('DFBA_CONTROL_AUDIT',PACKAGE.parents[1]/'local_verification/control_balances'))
HERE.mkdir(parents=True,exist_ok=True)
from surrogate_model import load_models
load=lambda p:json.loads(p.read_text())
checks=[]
def check(name,ok,detail=None): checks.append(dict(name=name,passed=bool(ok),detail=detail))
p=load(ROOT/'protocol.json');lock=load(ROOT/'results/selection_lock.json')
models=load_models(ROOT/'results/locked_models.json')
splits={s:np.load(ROOT/'data'/f'{s}.npz') for s in ['training','validation','test']}
conditions={s:{tuple(x) for x in p['conditions'][s]} for s in splits}
check('Condition groups pairwise disjoint',all(not conditions[a]&conditions[b] for a,b in [('training','validation'),('training','test'),('validation','test')]))
check('All split dimensions and labels',all(splits[s]['X'].shape==(n,9) and splits[s]['Y'].shape==(n,2) and np.isfinite(splits[s]['Y']).all() for s,n in [('training',1152),('validation',192),('test',256)]))
metrics=[]
for name,model in models.items():
 tr=splits['training'];te=splits['test'];va=splits['validation']
 check(name+' input scaler fitted to training only',np.allclose(np.asarray(model.record['input_mean']),tr['X'].mean(0),atol=1e-14,rtol=0) and np.allclose(np.asarray(model.record['input_scale']),tr['X'].std(0),atol=1e-14,rtol=0))
 check(name+' output scaler fitted to training only',np.allclose(np.asarray(model.record['output_mean']),tr['Y'].mean(0),atol=1e-14,rtol=0) and np.allclose(np.asarray(model.record['output_scale']),tr['Y'].std(0),atol=1e-14,rtol=0))
 pred=model.predict(te['X']);saved=np.load(ROOT/'data'/f'{name}_test_predictions.npz')
 check(name+' locked predictions equal recorded array',np.array_equal(pred,saved['prediction']) and np.array_equal(te['Y'],saved['truth']))
 selection=lock['selection'][name];vscore=np.mean(((model.predict(va['X'])-va['Y'])/np.asarray(model.record['output_scale']))**2)
 check(name+' selected by validation minimum',selection['selected']['validation_joint_standardized_mse']==min(r['validation_joint_standardized_mse'] for r in selection['candidates']) and abs(vscore-selection['selected']['validation_joint_standardized_mse'])<1e-14)
 for k,target in enumerate(['log_ratio','log_biomass_amount','biomass_amount_gDW']):
  truth=te['Y'][:,k] if k<2 else np.exp(te['Y'][:,1]);pr=pred[:,k] if k<2 else np.exp(pred[:,1]);e=pr-truth
  metrics.append(dict(method=name,target=target,n=len(truth),mse=float(np.mean(e**2)),rmse=float(np.sqrt(np.mean(e**2))),r_squared=float(1-sum(e**2)/sum((truth-truth.mean())**2)),mae=float(np.mean(abs(e)))))
check('MLP architecture 9-15-2 has 182 parameters',sum(np.asarray(x).size for x in models['mlp'].record['weights']+models['mlp'].record['biases'])==182)
rows=[load(x) for x in sorted((ROOT/'results/policies').glob('*.json'))]
methods=['direct_single_start','direct_multistart','database_warmstart','quadratic_ridge','mlp']
summary=load(ROOT/'results/summary.json');stats=[]
for m in methods:
 rr=[r for r in rows if r['method']==m];sm=next(z for z in summary['method_summary'] if z['method']==m)
 check(m+' counts and times aggregate exactly',sum(r['total_mechanistic_attempts'] for r in rr)==sm['total_actual_calls'] and abs(sum(r['online_total_seconds'] for r in rr)-sm['sum_online_seconds'])<1e-10)
 stats.append(dict(method=m,runs=len(rr),all_feasible=all(r['tight_verification']['feasible'] for r in rr),total_evaluations=sum(r['total_mechanistic_attempts'] for r in rr),median_evaluations=float(np.median([r['total_mechanistic_attempts'] for r in rr])),median_first_feasible=float(np.median([r['first_feasible_ordinary_evaluation'] for r in rr])),online_seconds=sm['sum_online_seconds'],offline_seconds=sm['offline_seconds_conservative_full_development_bundle'],total_seconds=sm['total_prespecified_seconds_for_recorded_36_decisions']))
# Independently code amounts, with analytic volume and integrated glucose consumption.
table=load(PACKAGE/'data/flux_table_eps1e-06.json')
curves=[]
for sp in ['SC','KP']:
 rr=table[sp]['rows'];curves.append(PchipInterpolator([q['S'] for q in rr],[[q['growth'],q['glucose_flux']] for q in rr],axis=0))
def integrate(row):
 initial=np.array(row['initial']);u=np.array(row['controls']);state=np.r_[initial[0]*initial[1:],0.];maxS=initial[3];minS=initial[3];max_res=0.
 for j,feed in enumerate(u):
  t0=2.*j;t1=t0+2.;vstart=initial[0]+2*sum(u[:j])
  def rhs(t,m):
   vol=vstart+feed*(t-t0);S=max(m[2]/vol,0.);rates=np.array([f(S) for f in curves])*min(1.,S/1e-5);uptake=-.18*(rates[0,1]*m[0]+rates[1,1]*m[1])
   return [rates[0,0]*m[0],rates[1,0]*m[1],feed*100-uptake,uptake]
  sol=solve_ivp(rhs,(t0,t1),state,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.125)
  assert sol.success
  vols=vstart+feed*(sol.t-t0);Ss=sol.y[2]/vols;maxS=max(maxS,float(max(Ss)));minS=min(minS,float(min(Ss)))
  fed=100*(vols-initial[0]);balance=initial[0]*initial[3]+fed-sol.y[2]-sol.y[3];max_res=max(max_res,float(max(abs(balance))))
  state=sol.y[:,-1]
 V=initial[0]+2*sum(u);terminal=np.r_[V,state[:3]/V];saved=np.array(row['tight_verification']['terminal']);z=np.log(state[0]/state[1]);B=sum(state[:2]);feas=abs(z-np.log(row['target_ratio']))<=np.log(1.01) and V<=1.8+1e-8
 return dict(policy=f"{row['case_id']}_{row['seed']}_{row['method']}",terminal=terminal.tolist(),biomass_gDW=float(B),max_terminal_difference=float(max(abs(terminal-saved))),relative_biomass_difference=float(abs(B-row['tight_verification']['biomass_amount_gDW'])/B),log_ratio_difference=float(abs(z-row['tight_verification']['z'])),glucose_balance_error_g=max_res,min_glucose_g_L=minS,max_glucose_g_L=maxS,net_biomass_glucose_yield_gDW_g=float((B-.001)/state[3]),feasibility_agrees=bool(feas==row['tight_verification']['feasible']))
started=time.perf_counter();amounts=[integrate(r) for r in rows]
check('All 180 independent amount integrations retain feasibility',all(r['feasibility_agrees'] for r in amounts))
check('Amount/concentration biomass differences below 1e-6 relative',max(r['relative_biomass_difference'] for r in amounts)<1e-6)
remaining_margin=min(np.log(1.01)-r['tight_verification']['log_ratio_error'] for r in rows)
check('Formulation differences less than 1 percent of smallest acceptance margin',max(r['log_ratio_difference'] for r in amounts)<.01*remaining_margin,dict(remaining_margin=float(remaining_margin),comparison='Absolute difference divided by worst remaining final-acceptance margin; initial arbitrary 1e-6 screen and subsequent convergence study retained separately'))
refinement=load(HERE/'tolerance_convergence.json')
check('Three largest-discrepancy policies agree below 1e-7 at refined tolerances',all(r['final_formulation_difference']<1e-7 and r['last_concentration_change']<1e-7 and r['last_amount_change']<1e-7 for r in refinement))
check('Glucose amount balance below 1e-9 g',max(r['glucose_balance_error_g'] for r in amounts)<1e-9)
check('All accepted substrate states in response-table domain',all(0<=r['min_glucose_g_L']<=r['max_glucose_g_L']<=100 for r in amounts))
sens=load(ROOT/'results/initial_condition_sensitivity.json')['cases']
check('Sensitivity coverage 240 and no policy passes every corner',len(sens)==240 and all(sum(q['feasible'] for q in sens if q['case_id']==r['case_id'] and q['method']==r['method'])<4 for r in rows if r['seed']==3101))
# Paired case-and-seed contrasts; do not compare unmatched best runs.
paired=[]
for r in rows:
 if r['method']=='mlp':
  d=next(q for q in rows if q['method']=='direct_single_start' and q['case_id']==r['case_id'] and q['seed']==r['seed']);paired.append(100*(r['tight_verification']['biomass_amount_gDW']/d['tight_verification']['biomass_amount_gDW']-1))
report=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',date='2026-10-22',scope='Independent data/selection/arithmetic audit and 180 fresh amount-balance integrations; no optimisation or direct KKT campaign rerun',checks=checks,passed=sum(c['passed'] for c in checks),total=len(checks),prediction_metrics=metrics,method_statistics=stats,independent_amount_summary=dict(seconds=time.perf_counter()-started,max_terminal_difference=max(r['max_terminal_difference'] for r in amounts),max_relative_biomass_difference=max(r['relative_biomass_difference'] for r in amounts),max_log_ratio_difference=max(r['log_ratio_difference'] for r in amounts),maximum_glucose_balance_error_g=max(r['glucose_balance_error_g'] for r in amounts),yield_range_gDW_g=[min(r['net_biomass_glucose_yield_gDW_g'] for r in amounts),max(r['net_biomass_glucose_yield_gDW_g'] for r in amounts)]),paired_neural_biomass_difference_percent=dict(min=min(paired),max=max(paired),median=float(np.median(paired))),amount_records=amounts,source_hashes={str(q.relative_to(PACKAGE)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [ROOT/'protocol.json',ROOT/'results/selection_lock.json',ROOT/'results/summary.json',PACKAGE/'data/flux_table_eps1e-06.json']})
(HERE/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['amount_records','source_hashes','checks']},indent=2))
if report['status']!='PASS':raise SystemExit('Audit failed')
