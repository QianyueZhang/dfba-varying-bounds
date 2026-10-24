"""Audit original independent/shared-resource horizons without unsupported continuation.

Physical source parameters and active ATPM are retained. Direct KKT solutions
replace clipped flux initialisation. A small positive LP-growth guard marks the
last reported active-domain state. This script does NOT invent starvation,
quiescent reactivation, active fractions or adopt truncated replacement figures.
"""
from pathlib import Path
import argparse,sys,time,json,platform,hashlib
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research_completion/scripts/python_migration'))
sys.path.insert(0,str(ROOT/'research_completion/scripts'))
from interaction_pilot import Core,CASES,BASE
from primal_dual import solve as solve_pd
OUT=ROOT/'research_completion/runs/python_consistency/independent_shared'
MU=1e-8;TOL=1e-11;ENTRY=1e-5;F=.1;V0=5.;MW=.18

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Model(Core):
 def __init__(self,folder,species,vmax):
  super().__init__(folder,species,vmax)
  self.guard_G=brentq(lambda g:-self.lp(g).fun-ENTRY,self.critical_G+1e-9,1.,xtol=2e-14)
  self.physical_zero_growth_G=brentq(lambda g:-self.lp(g).fun,self.critical_G+1e-9,1.,xtol=2e-14)
 def flux(self,G):
  self.calls+=1
  try:sol=solve_pd(self,G,mu=MU,tol=TOL,warm=self.warm)
  except RuntimeError:
   self.retries+=1;sol=solve_pd(self,G,mu=MU,tol=TOL,warm=None)
  self.warm=sol[:4];v=sol[0];d=sol[-1]
  self.newton_steps+=d['newton_steps'];self.max_eq=max(self.max_eq,d['equality_residual'])
  self.max_raw_eq=max(self.max_raw_eq,float(np.max(np.abs(self.raw_A@v))))
  self.max_pd=max(self.max_pd,d['primal_dual_residual']);self.max_explicit=max(self.max_explicit,d['stationarity_residual'])
  self.min_slack=min(self.min_slack,d['minimum_slack']);return v

def make_models(case):
 species=case.get('species_indices',list(range(len(case['vmax']))))
 return [Model(BASE/case['folder'],'S'+str(species[i]+1),x)for i,x in enumerate(case['vmax'])]

def integrate(case,amount=False):
 mods=make_models(case);n=len(mods);ng=len(case['initial_substrate']);clamped=0
 init=np.r_[np.ones(n)*V0,np.array(case['initial_substrate'])*V0,np.zeros(n)] if amount else np.r_[V0,np.ones(n),case['initial_substrate']]
 def decode(t,y):
  if amount:return V0+F*t,y[:n]/(V0+F*t),y[n:n+ng]/(V0+F*t)
  return y[0],y[1:1+n],y[1+n:]
 def rhs(t,y):
  nonlocal clamped
  V,X,G=decode(t,y);vs=[]
  for i,m in enumerate(mods):
   g=G[0 if case['shared']else i]
   if g<m.guard_G:clamped+=1
   # Event bracketing extension only. No retained active state lies below guard.
   vs.append(m.flux(max(float(g),m.guard_G)))
  vs=np.asarray(vs);growth=vs[:,12]*X;exchange=MW*vs[:,27]*X
  met=np.array([exchange.sum()])if case['shared']else exchange
  if amount:return np.r_[growth*V,F*np.array(case['feed_substrate'])+met*V,exchange*V]
  return np.r_[F,growth-F/V*X,F/V*(np.array(case['feed_substrate'])-G)+met]
 events=[]
 for i,m in enumerate(mods):
  def guard(t,y,i=i,m=m):return decode(t,y)[2][0 if case['shared']else i]-m.guard_G
  guard.terminal=True;guard.direction=-1;events.append(guard)
 start=time.perf_counter()
 sol=solve_ivp(rhs,(0,case['duration_h']),init,method='DOP853'if amount else'RK45',rtol=1e-10 if amount else 1e-8,atol=1e-12 if amount else 1e-10,max_step=.005 if amount else .01,events=events,dense_output=True)
 if not sol.success:raise RuntimeError(sol.message)
 return sol,mods,decode,{'formulation':'amounts with analytic volume'if amount else'concentrations with integrated volume','method':'DOP853'if amount else'RK45','rtol':1e-10 if amount else 1e-8,'atol':1e-12 if amount else 1e-10,'max_step_h':.005 if amount else.01,'terminal_time_h':float(sol.t[-1]),'complete_horizon':bool(sol.t[-1]>=case['duration_h']-1e-10),'event_times':[x.tolist()for x in sol.t_events],'RHS_evaluations':sol.nfev,'event_bracketing_below_guard_evaluations':clamped,'elapsed_s':time.perf_counter()-start,'models':[m.metrics()for m in mods]}

def run(key,case):
 out=OUT/key;out.mkdir(parents=True,exist_ok=True);n=len(case['vmax']);ng=len(case['initial_substrate'])
 print('START',key,flush=True)
 s1,m1,d1,r1=integrate(case,False);s2,m2,d2,r2=integrate(case,True)
 end=min(s1.t[-1],s2.t[-1]);ts=np.linspace(0,end,501)
 def physical(sol,decode):return np.array([np.r_[decode(t,sol.sol(t))[0],decode(t,sol.sol(t))[1],decode(t,sol.sol(t))[2]]for t in ts])
 y1=physical(s1,d1);y2=physical(s2,d2);diff=np.max(abs(y2-y1),axis=0);mods=make_models(case)
 vv=[];lpchecks=[];bound_excess=0.
 for j,(t,row)in enumerate(zip(ts,y2)):
  pair=[]
  for i,m in enumerate(mods):
   G=row[1+n+(0 if case['shared']else i)];v=m.flux(G);pair.append(v);l,u=m.bounds(G)
   bound_excess=max(bound_excess,float(np.max(l-v)),float(np.max(v-u)))
   if j%50==0:lpchecks.append({'time_h':float(t),'species':i+1,'LP_growth':float(-m.lp(G).fun),'KKT_growth':float(v[12]),'objective_gap':float(-m.lp(G).fun-v[12]),'bound_2nmu':190*MU})
  vv.append(pair)
 vv=np.array(vv);amounts=s2.sol(ts).T
 expected=V0*np.array(case['initial_substrate'])+ts[:,None]*F*np.array(case['feed_substrate'])
 expected+=amounts[:,n+ng:].sum(axis=1,keepdims=True)if case['shared']else amounts[:,n+ng:]
 balances=np.max(abs(amounts[:,n:n+ng]-expected),axis=0)
 terminal=y2[-1];V=terminal[0];X=terminal[1:1+n];G=terminal[1+n:];last=vv[-1]
 demand=-MW*last[:,27]*X*V
 boundaries=[]
 for i,m in enumerate(mods):
  pool=0 if case['shared']else i;required=float(demand.sum())if case['shared']else float(demand[i]);available=F*(case['feed_substrate'][pool]-G[pool])
  boundaries.append({'species':i+1,'ATPM_lower_bound':float(m.lo[10]),'minimum_glucose_for_relaxed_active_LP_g_L':m.critical_G,'zero_LP_growth_glucose_g_L':m.physical_zero_growth_G,'positive_growth_guard_g_L':m.guard_G,'guard_maximum_growth_h_inv':ENTRY,'terminal_pool_g_L':float(G[pool]),'gross_metabolic_glucose_demand_g_h':required,'feed_available_at_constant_concentration_g_h':float(available),'pool_concentration_derivative_g_L_h':float((available-required)/V),'terminal_species_glucose_uptake_mmol_gDW_h':float(-last[i,27])})
 verified=bool(max(diff)<2e-6 and bound_excess<=1e-10 and all(m.max_raw_eq<1e-9 and m.max_pd<2e-11 and m.min_slack>0 for m in mods)and min(y2[:,1:].ravel())>=0 and max(balances)<1e-9 and all(-1e-10<x['objective_gap']<x['bound_2nmu']+1e-10 for x in lpchecks))
 names=['time_h','V_L']+[f'X{i+1}_gDW_L'for i in range(n)]+[f'G{i+1}_g_L'for i in range(ng)]+[f'growth{i+1}_h_inv'for i in range(n)]+[f'glucose_exchange{i+1}_mmol_gDW_h'for i in range(n)]
 np.savetxt(out/'trajectory.csv',np.c_[ts,y2,vv[:,:,12],vv[:,:,27]],delimiter=',',header=','.join(names),comments='')
 np.savez_compressed(out/'trajectory.npz',time_h=ts,concentrations=y2,flux=vv,amount_state=amounts,primary_concentrations=y1)
 report={'case':key,'case_definition':case,'original_requested_horizon_h':case['duration_h'],'reported_terminal_time_h':float(end),'full_horizon_completed':r1['complete_horizon']and r2['complete_horizon'],'terminal_state_V_X_G':terminal.tolist(),'minimum_state_V_X_G':np.min(y2,axis=0).tolist(),'integrations':[r1,r2],'numerical':{'mu':MU,'source_mu':1e-6,'source_finite_bound_margin':1e-4,'KKT_tolerance':TOL,'positive_LP_growth_guard':ENTRY,'reason_for_mu':'1e-8 limits the 2nmu gap to1.9e-6, below the1e-5 positive-growth event guard. This is a stated diagnostic numerical setting, not a source-equivalence assertion.'},'boundary_supply_and_demand':boundaries,'validation':{'independent_formulation_max_difference_V_X_G':diff.tolist(),'event_time_difference_h':abs(float(s1.t[-1])-float(s2.t[-1])),'integrated_pool_balance_max_error_g':balances.tolist(),'analytic_volume_max_error_L':float(np.max(abs(y2[:,0]-V0-F*ts))),'maximum_active_bound_excess':bound_excess,'sampled_model_metrics':[m.metrics()for m in mods],'LP_gap_checks':lpchecks,'passed':verified},'adoption_ready':False,'scope':'Active-model feasibility audit on original requested horizon. Terminal event stops before loss of active ATPM feasibility. No post-event curve, dormant-cell model or shortened-horizon replacement is proposed.','source_files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for p in(BASE/case['folder']).iterdir()if p.suffix in['.m','.mat','.txt']],'script_sha256':sha(Path(__file__)),'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()}}
 (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
 if not verified:raise RuntimeError('Validation failure '+key)
 print('DONE',key,end,'full',report['full_horizon_completed'],'terminal',terminal,'difference',diff.max(),flush=True)
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',choices=list(CASES)+['matched_controls']);args=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 if args.case=='matched_controls':
  base=CASES['shared_competition']
  for i in range(2):
   case=dict(base);case['vmax']=[base['vmax'][i]];case['species_indices']=[i];case['shared']=False;case['figures']=[]
   case['parameter_basis']='Single-species validation control with identical initial glucose, inlet glucose, biomass, volume, feed, source bounds and uptake parameters to the corresponding shared-resource population.'
   run('matched_control_species'+str(i+1),case)
 else:
  for key in [args.case]if args.case else CASES:run(key,CASES[key])
