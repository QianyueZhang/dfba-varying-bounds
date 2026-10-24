"""Independent checks of saved 72 h study arrays; never runs dynamics."""
from pathlib import Path
import json,hashlib,sys,datetime
import numpy as np
from scipy.sparse import csc_matrix
from scipy.io import loadmat
import benchmark as b
OUT=Path(__file__).resolve().parent;ROOT=b.ROOT
LABELS=['nominal_light','nominal_dark','nominal_light_parameter_refined','nominal_dark_parameter_refined','nominal_light_ode_refined','loading_002_light','loading_005_light']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
models={}
for name,file in [('SC','iND750.mat'),('RC','iRC1080.mat')]:
 raw=next(v for k,v in loadmat(ROOT/'data/models/cstr'/file,simplify_cells=True).items()if not k.startswith('__'))
 models[name]={'S':csc_matrix(raw['S']),'ids':{str(v):i for i,v in enumerate(raw['rxns'])}}
checks=[];records={};arrays={}
def check(name,ok,details=None):checks.append({'name':name,'passed':bool(ok),'details':details})
for label in LABELS:
 if not (OUT/(label+'.json')).exists():continue
 r=json.loads((OUT/(label+'.json')).read_text());d=np.load(OUT/(label+'.npz'));arrays[label]=d;t=d['time_h'];y=d['states'];duration=r['protocol']['horizon_h'];initial=np.array(list(r['protocol']['initial_state'].values()));inlet=np.array(list(r['protocol']['inlet_mmol_L'].values()));D=r['protocol']['dilution_h_inv'];eps=r['numerics']['epsilon']
 check(label+': complete monotone 0.1 h outputs',len(t)==int(round(duration/.1))+1 and t[0]==0 and t[-1]==duration and np.all(np.diff(t)>0) and np.allclose(np.diff(t),.1,rtol=0,atol=2e-14))
 check(label+': nonnegative and exact initial/terminal state',np.all(y>=0) and np.array_equal(y[0],initial) and np.max(abs(y[-1]-np.array(list(r['terminal'].values()))))<1e-14)
 check(label+': run-recorded model/source hashes unchanged',all(sha(ROOT/p)==h for p,h in r['protocol']['source_hashes'].items()))
 check(label+': finite outputs and nonnegative internal ODE states',all(np.all(np.isfinite(d[k])) for k in d.files) and np.all(np.array(r['minimum_raw_rhs_states'])>=0))
 rates={};detail={'label':label,'case':r['case'],'duration_h':duration,'initial':r['protocol']['initial_state'],'terminal':r['terminal'],'biomass_change_percent':(100*(y[-1,1:3]/y[0,1:3]-1)).tolist(),'washout_reference':(y[0,1:3]*np.exp(-D*duration)).tolist(),'substrate_depletion_percent':(100*(1-y[-1,3:5]/y[0,3:5])).tolist(),'minima':y.min(axis=0).tolist(),'maxima':y.max(axis=0).tolist(),'growth':{},'full_constraint_check':{}}
 for sp in ['SC','RC']:
  key=sp.lower();v=d[key+'_flux'];lo=d[key+'_lower'];up=d[key+'_upper'];ids=models[sp]['ids'];x=y[:,1 if sp=='SC'else 2];bio=ids['BIOMASS_SC4_bal'if sp=='SC'else'BIOMASS_Chlamy_auto'];g=v[:,bio]
  eq=float(np.max(abs(models[sp]['S']@v.T)));sl=float(min(np.min(v-lo),np.min(up-v)));detail['full_constraint_check'][sp]={'raw_stoichiometric_residual':eq,'minimum_slack':sl}
  check(label+'/'+sp+': complete raw stoichiometry and scalar bounds',eq<1e-9 and sl>0,detail['full_constraint_check'][sp])
  detail['growth'][sp]={'minimum':float(g.min()),'maximum':float(g.max()),'fraction_times_above_dilution':float(np.mean(g>D)),'mean_from_biomass_identity':float(np.log(x[-1]/x[0])/duration+D),'quadrature_identity_error':float(np.log(x[-1]/x[0])+D*duration-np.trapezoid(g,t))}
  ids_exchange={'acetate':'EX_ac_e','oxygen':'EX_o2_e','carbon_dioxide':'EX_co2_e','ethanol':'EX_etoh_e'}
  if sp=='SC':ids_exchange['glucose']='EX_glc__D_e'
  rates[sp]={k:v[:,ids[j]]*x for k,j in ids_exchange.items()}
  if sp=='RC':
   rates[sp]['photosynthetic_oxygen']=v[:,ids['DM_o2D_u']]*x;rates[sp]['net_oxygen']=rates[sp]['oxygen']+rates[sp]['photosynthetic_oxygen']
   detail['algal_net_oxygen_specific_range']=[float(np.min((rates[sp]['net_oxygen']/x))),float(np.max((rates[sp]['net_oxygen']/x)))]
  detail[sp+'_exchange_integrals']={k:{'net_secretion':float(np.trapezoid(vv,t)),'gross_uptake':float(np.trapezoid(np.maximum(-vv,0),t)),'gross_secretion':float(np.trapezoid(np.maximum(vv,0),t))}for k,vv in rates[sp].items()}
 integ=d['integrals'][-1];balances=y[-1,3:]-y[0,3:]-(D*inlet*duration-integ[:5]+integ[5:10]+np.r_[0,0,integ[10:12],0]);trap=[]
 for j,key in enumerate(b.SOLUTES):
  cell=rates['SC'].get(key,0)+rates['RC'].get('net_oxygen'if key=='oxygen'else key,0);gas=b.KLA*(inlet[j]-y[:,j+3])if j in [2,3]else 0
  trap.append(float(y[-1,j+3]-y[0,j+3]-np.trapezoid(D*(inlet[j]-y[:,j+3])+cell+gas,t)))
 detail['integrated_balance_residual_mmol_L']=balances.tolist();detail['sampled_quadrature_residual_mmol_L']=trap
 check(label+': integrated inventory balances',np.max(abs(balances))<1e-10,balances.tolist())
 check(label+': independent saved-flux quadrature',max(abs(np.array(trap)))<1e-4,trap)
 flux=d['rc_flux'][:,models['RC']['ids']['PRISM_solar_litho']];cap=np.array([54*b.light_fraction(ti)if r['case']=='light'else 0 for ti in t]);throughput=flux*y[:,0]*y[:,2];allow=eps*y[:,0]*y[:,2]
 detail['photon_budget']={'incident_integral_mmol':float(np.trapezoid(cap,t)),'throughput_integral_mmol':float(np.trapezoid(throughput,t)),'max_excess_mmol_h':float(np.max(throughput-cap)),'relaxation_allowance_max_mmol_h':float(np.max(allow)),'max_budget_residual_after_allowance':float(np.max(throughput-cap-allow))}
 check(label+': finite reactor photon budget',np.max(throughput-cap-allow)<1e-9,detail['photon_budget'])
 detail['oxygen_saturation_ratio_range']=[float(y[:,5].min()/b.O_SAT),float(y[:,5].max()/b.O_SAT)];detail['alga_net_oxygen_always_negative']=bool(np.all(rates['RC']['net_oxygen']<0))
 detail['maximum_solver_equality']=r['max_equality_residual'];detail['maximum_solver_primal_dual']=r['max_primal_dual_residual'];detail['maximum_sampled_LP_growth_gap']=max(abs(x['LP_growth_gap'])for x in r['sampled_LP_checks'])
 check(label+': solver and LP growth accuracy',r['max_equality_residual']<1e-9 and r['max_primal_dual_residual']<1.01e-10 and detail['maximum_sampled_LP_growth_gap']<1e-5)
 detail['maximum_effective_relative_complementarity']=r['max_effective_relative_complementarity_residual']
 check(label+': positive slacks and relative complementarity',r['minimum_slack']>0 and r['max_effective_relative_complementarity_residual']<=2.01e-8)
 check(label+': LP growth differences respect finite-barrier objective bound',all(-2e-9<=x['LP_growth_gap']<=2*(1266 if x['organism']=='SC' else 2191)*r['numerics']['mu']+2e-9 for x in r['sampled_LP_checks']))
 detail['daily_biomass']=[{'time_h':float(ti),'X_SC':float(y[np.argmin(abs(t-ti)),1]),'X_RC':float(y[np.argmin(abs(t-ti)),2])}for ti in range(0,int(duration)+1,24)]
 records[label]=detail
refinements={}
for target,base in [('nominal_light_parameter_refined','nominal_light'),('nominal_dark_parameter_refined','nominal_dark'),('nominal_light_ode_refined','nominal_light')]:
 if target not in arrays or base not in arrays:continue
 difference=arrays[target]['states']-arrays[base]['states'];r={'max_biomass_abs_gDW_L':float(np.max(abs(difference[:,1:3]))),'terminal_biomass_abs_gDW_L':float(np.max(abs(difference[-1,1:3]))),'max_solute_abs_mmol_L':float(np.max(abs(difference[:,3:]))),'terminal_solute_abs_mmol_L':float(np.max(abs(difference[-1,3:])))};refinements[target]=r
 check(target+': trajectory refinement small on stated physical scales',r['max_biomass_abs_gDW_L']<2.5e-5 and r['max_solute_abs_mmol_L']<5e-4,r)
comparison={}
if 'nominal_light'in arrays and 'nominal_dark'in arrays:
 L=arrays['nominal_light']['states'];K=arrays['nominal_dark']['states'];comparison={'light_vs_dark_terminal_biomass_percent':(100*(L[-1,1:3]/K[-1,1:3]-1)).tolist(),'max_oxygen_difference_mmol_L':float(np.max(L[:,5]-K[:,5])),'terminal_ethanol_light_dark':L[-1,7:].tolist()+K[-1,7:].tolist(),'algal_gross_oxygen_uptake_reduction_percent':float(100*(1-records['nominal_light']['RC_exchange_integrals']['net_oxygen']['gross_uptake']/records['nominal_dark']['RC_exchange_integrals']['net_oxygen']['gross_uptake'])),'algal_integrated_net_oxygen_consumption_reduction_percent':float(100*(1-records['nominal_light']['RC_exchange_integrals']['net_oxygen']['net_secretion']/records['nominal_dark']['RC_exchange_integrals']['net_oxygen']['net_secretion']))}
report={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Independent complete raw stoichiometry, bounds, inventories, photon budgets and numerical refinement from saved trajectories; no new dynamic integration.','predeclared_plan':'STUDY_PLAN.json','runs':records,'comparison':comparison,'refinements':refinements,'checks':checks,'all_passed':len(records)==len(LABELS) and all(c['passed']for c in checks),'available_runs':len(records),'expected_runs':len(LABELS),'missing_runs':[n for n in LABELS if n not in records],'sha256':{str(p.relative_to(ROOT)):sha(p)for p in OUT.iterdir() if p.suffix in ['.py','.npz','.csv','.json'] and p.name not in ['AUDIT.json','adoption_validation.json']}}
(OUT/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'available':len(records),'passed':report['all_passed'],'checks':len(checks),'missing':report['missing_runs'],'failed':[c for c in checks if not c['passed']],'comparison':comparison,'refinements':refinements},indent=2))
