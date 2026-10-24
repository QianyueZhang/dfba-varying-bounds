"""Direct-KKT audit of selected source product-inhibition cases19/20.

Retains source physical parameters and fixed ATP maintenance. No quiescence,
maintenance relaxation or extended post-infeasibility trajectory is introduced.
A specified positive distance to the exact LP-feasibility boundary is the
terminal numerical guard; it is never called the exact boundary itself.
"""
from pathlib import Path
import sys,json,time,hashlib,platform,argparse
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from scipy.optimize import linprog
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research_completion/scripts'))
from primal_dual import solve as solve_pd
OUT=ROOT/'research_completion/runs/python_consistency/product_inhibition'
SOURCES={
 'separate':ROOT/'data/models/product_inhibition'}
EPS=1e-4;MU=1e-6;KI=.1;F=.1;V0=5.;MWG=.18;MWE=.046

def flat(v):return np.asarray(v.toarray() if hasattr(v,'toarray') else v,float).ravel()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class Model:
 def __init__(self,case,species,tol=1e-10,inhibition=True):
  self.case=case;self.species=species;self.name=case+'_'+species;self.bio=12;self.g=27;self.e=23;self.tol=tol;self.inhibition=inhibition
  self.source=SOURCES[case]
  self.saved=loadmat(self.source/f'{species}_Model.mat',simplify_cells=True)['Struct']
  raw=loadmat(self.source/'e_coli_core.mat',simplify_cells=True)['e_coli_core']
  assert raw['rxns'][self.e]=='EX_etoh_e' and raw['rxns'][self.g]=='EX_glc__D_e'
  self.rawA=csc_matrix(raw['S']);self.A=csc_matrix(self.saved['A']);self.b=flat(self.saved['b']);self.c=flat(self.saved['c']);self.lo=flat(self.saved['lb']);self.up=flat(self.saved['ub'])
  self.level=1.;self.warm=None;self.cache=None;self.required_cache={};self.calls=0;self.retries=0;self.maxeq=0.;self.maxpd=0.;self.minfree=np.inf
 def uptake(self,env):
  G,E=env;q=10*G/(1+G)
  return q/(1+E/KI) if self.species=='S2' and self.inhibition else q
 def bounds(self,env):
  l=self.lo.copy();u=self.up.copy()
  l[self.g]=max(-self.uptake(env),-10-EPS)
  if self.species=='S1':l[self.e]=self.level-EPS
  return np.minimum(l,u-2*EPS),u
 def lp(self,env):
  key=tuple(env)+(self.level,)
  if self.cache and self.cache[0]==key:return self.cache[1]
  l,u=self.bounds(env)
  ans=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=list(zip(l,u)),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  self.cache=(key,ans);return ans
 def required(self):
  # Exact source-model feasibility threshold: closest-to-zero feasible glucose exchange.
  # Other bounds are fixed within each donor secretion phase.
  key=self.level if self.species=='S1' else 0.
  if key not in self.required_cache:
   l,u=self.bounds((20.,0.));l[self.g]=-10-EPS
   c=np.zeros(95);c[self.g]=-1.
   ans=linprog(c,A_eq=self.A,b_eq=self.b,bounds=list(zip(l,u)),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
   assert ans.success
   self.required_cache[key]=(-float(ans.x[self.g]),ans.x)
  return self.required_cache[key][0]
 def margin(self,env):return max(self.uptake(env),EPS)-self.required()
 def flux(self,env):
  self.calls+=1
  try:sol=solve_pd(self,env,mu=MU,tol=self.tol,warm=self.warm)
  except RuntimeError:
   self.retries+=1;sol=solve_pd(self,env,mu=MU,tol=self.tol,warm=None)
  self.warm=sol[:4];v=sol[0];d=sol[-1]
  self.maxeq=max(self.maxeq,float(abs(self.rawA@v).max()));self.maxpd=max(self.maxpd,d['primal_dual_residual']);self.minfree=min(self.minfree,d['minimum_slack'])
  return v
 def initial_audit(self):
  env=(20. if self.case=='shared' else 10.,0.)
  v=flat(self.saved['flux']);lam=flat(self.saved['lamd']);sl=v-self.lo;su=self.up-v
  l,u=self.bounds(env);clipped=np.maximum(l+EPS,np.minimum(u-EPS,v))
  direct=self.flux(env);lp=self.lp(env)
  return {'stored_initial':flat(self.saved['INITIAL']).tolist(),'stored_uptake':flat(self.saved['uptk']).tolist(),'raw_stoichiometric_residual':float(abs(self.rawA@v).max()),'stationarity_stored_bounds':float(abs(-self.c+self.A.T@lam-MU/sl+MU/su).max()),'stored_minimum_slack':float(min(sl.min(),su.min())),'effective_source_RHS_clipped_stoichiometric_residual':float(abs(self.rawA@clipped).max()),'effective_source_RHS_growth':float(clipped[self.bio]),'consistent_direct_growth':float(direct[self.bio]),'matched_LP_max_growth':float(-lp.fun),'LP_growth_gap_of_clipped_saved_flux':float(-lp.fun-clipped[self.bio]),'active_ATPM_lower_bound':float(self.lo[10]),'source_E_lower_bound':float(l[23])}

def integrate(case,guard=1e-4,rtol=1e-8,atol=1e-10,max_step=.01,tol=1e-10,inhibition=True,complete_ethanol_balance=False):
 mods=[Model(case,s,tol,inhibition)for s in ['S1','S2']]
 init=[m.initial_audit() for m in mods]
 # Amounts: X1,X2,G1,G2,E; cumulative growth1/2, glucose exchange1/2, ethanol exchange1/2.
 y0=np.r_[[5,5,100,0,0]if case=='shared'else[5,5,50,50,0],np.zeros(6)].astype(float)
 t0=0.;level=1.;segments=[];events=[];started=time.perf_counter()
 feed=F*(np.array([20.,0.])if case=='shared'else np.array([10.,10.]))
 def envs(t,y):
  V=V0+F*t;return [(y[2]/V,y[4]/V),(y[2 if case=='shared'else 3]/V,y[4]/V)]
 def rhs_for_phase(t,y,allow_guard=True):
  env=envs(t,y);vv=[]
  for i,m in enumerate(mods):
   if allow_guard and m.margin(env[i])<guard*.2:vv.append(np.zeros(95))
   else:vv.append(m.flux(env[i]))
  growth=np.array([vv[i][12]*y[i]for i in range(2)])
  gg=np.array([MWG*vv[i][27]*y[i]for i in range(2)])
  ee=np.array([MWE*vv[i][23]*y[i]for i in range(2)])
  dg=feed+np.array([gg.sum(),0.]) if case=='shared' else feed+gg
  # Exactly the source E balance: only donor ethanol exchange contributes.
  return np.r_[growth,dg,ee.sum()if complete_ethanol_balance else ee[0],growth,gg,ee]
 while t0<10-1e-12:
  mods[0].level=level
  for m in mods:m.warm=None;m.cache=None
  funcs=[];labels=[]
  if level>0:
   threshold=7.5 if level==1 else 2.
   def switch(t,y):return envs(t,y)[0][0]-threshold
   switch.terminal=True;switch.direction=-1;funcs.append(switch);labels.append(('secretion',threshold))
  for i,m in enumerate(mods):
   def margin_event(t,y,i=i,m=m):return m.margin(envs(t,y)[i])-guard
   margin_event.terminal=True;margin_event.direction=-1;funcs.append(margin_event);labels.append(('feasibility_guard',i))
  sol=solve_ivp(rhs_for_phase,(t0,10.),y0,method='RK45',rtol=rtol,atol=atol,max_step=max_step,dense_output=True,events=funcs)
  assert sol.success,sol.message
  segments.append({'t0':t0,'tf':float(sol.t[-1]),'level':level,'sol':sol})
  t0=float(sol.t[-1]);y0=sol.y[:,-1]
  if t0>=10-1e-12:break
  j=next(j for j,x in enumerate(sol.t_events)if len(x));kind,value=labels[j]
  ev={'time_h':t0,'event':kind,'value':value,'concentrations':(y0[:5]/(V0+F*t0)).tolist(),'margins': [m.margin(e)for m,e in zip(mods,envs(t0,y0))]}
  events.append(ev);print(case,ev,flush=True)
  if kind=='feasibility_guard':break
  level-=.5
 # Stored sample/diagnostics with fresh KKT and independent LP at every saved time.
 tf=segments[-1]['tf'];tt=np.unique(np.r_[np.linspace(0,tf,401),[e['time_h']for e in events]])
 yy=[];vv=[];levels=[];maxgap=0.;maxbound=0.;minbio=np.inf;maxmissing=0.
 for t in tt:
  seg=next((s for s in segments if s['t0']<=t<s['tf']),segments[-1]);mods[0].level=seg['level']
  y=seg['sol'].sol(t);env=envs(t,y);pair=[]
  for i,m in enumerate(mods):
   m.warm=None;m.cache=None;v=m.flux(env[i]);lp=m.lp(env[i]);assert lp.success
   l,u=m.bounds(env[i]);maxbound=max(maxbound,float((l-v).max()),float((v-u).max()));maxgap=max(maxgap,float(-lp.fun-v[12]));minbio=min(minbio,float(v[12]));pair.append(v)
  yy.append(y);vv.append(pair);levels.append(seg['level'])
 yy=np.asarray(yy);vv=np.asarray(vv);levels=np.asarray(levels)
 expected=np.c_[5+yy[:,5],5+yy[:,6],(100 if case=='shared'else 50)+feed[0]*tt+yy[:,7]+(yy[:,8]if case=='shared'else 0),np.zeros_like(tt)if case=='shared'else 50+feed[1]*tt+yy[:,8],yy[:,9]+(yy[:,10]if complete_ethanol_balance else 0)]
 balance=np.max(abs(yy[:,:5]-expected),axis=0)
 terminal=[]
 for i,m in enumerate(mods):
  m.level=levels[-1]if i==0 else 1.;env=envs(tf,yy[-1])[i];l,u=m.bounds(env);crit=m.required();tests=[]
  for delta in [1e-5,0.,-1e-5]:
   ll=l.copy();ll[27]=-(crit+delta)
   lp=linprog(-m.c,A_eq=m.A,b_eq=m.b,bounds=list(zip(ll,u)),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
   tests.append({'capacity_minus_exact_required':delta,'LP_success':bool(lp.success),'LP_status':int(lp.status),'growth':float(-lp.fun)if lp.success else None})
  dy=rhs_for_phase(tf,yy[-1],False);V=V0+F*tf;gi=2 if i==0 or case=='shared'else 3;G,E=env;dG=(dy[gi]-G*F)/V;dE=(dy[4]-E*F)/V
  dq=10*dG/(1+G)**2
  if i==1 and inhibition:dq=dq/(1+E/KI)-10*G/(1+G)*dE/KI/(1+E/KI)**2
  margin=m.margin(env)
  terminal.append({'species':i+1,'required_glucose_capacity_mmol_gDW_h':crit,'available_capacity':m.uptake(env),'margin':margin,'margin_derivative_per_h':float(dq),'linearised_boundary_time_h':float(tf-margin/dq)if dq<0 else None,'LP_boundary_checks':tests})
 return {'case':case,'initialisation':init,'events':events,'full_10h':bool(tf==10),'terminal_time_h':tf,'terminal_concentrations':(yy[-1,:5]/(V0+F*tf)).tolist(),'terminal_volume_L':V0+F*tf,'feasibility_trigger':terminal,'numeric':{'mu':MU,'epsilon':EPS,'guard_flux_margin':guard,'rtol':rtol,'atol':atol,'max_step_h':max_step,'KKT_tolerance':tol,'method':'RK45'},'active_metrics':[{'max_raw_equality_residual':m.maxeq,'max_primal_dual_residual':m.maxpd,'minimum_slack':m.minfree,'KKT_calls':m.calls,'cold_retries':m.retries}for m in mods],'max_stored_bound_excess':maxbound,'max_stored_LP_growth_gap':maxgap,'theoretical_gap_bound':190*MU,'minimum_stored_biomass_flux':minbio,'tracked_balance_errors_g':balance.tolist(),'recipient_ethanol_net_exchange_omitted_by_source_g':float(yy[-1,10]),'maximum_omitted_recipient_ethanol_flux':float(abs(vv[:,1,23]).max()),'elapsed_s':time.perf_counter()-started},(tt,yy,vv,levels)

def main():
 global F
 p=argparse.ArgumentParser();p.add_argument('case',choices=['separate']);p.add_argument('--refined',action='store_true');p.add_argument('--flow',type=float,default=.1);p.add_argument('--disable-inhibition',action='store_true');p.add_argument('--complete-ethanol-balance',action='store_true');a=p.parse_args()
 F=a.flow
 variant=('refined'if a.refined else'primary')
 if F!=.1:variant='flow_'+str(F).replace('.','p')+'_'+variant
 if a.disable_inhibition:variant+='_reference'
 if a.complete_ethanol_balance:variant+='_balanced'
 dest=OUT/a.case/variant;dest.mkdir(parents=True,exist_ok=True)
 config=dict(guard=1e-6,rtol=1e-10,atol=1e-12,max_step=.005,tol=1e-11)if a.refined else{}
 try:r,data=integrate(a.case,inhibition=not a.disable_inhibition,complete_ethanol_balance=a.complete_ethanol_balance,**config)
 except Exception as e:
  (dest/'failure.json').write_text(json.dumps({'error':str(e),'config':config},indent=2));raise
 r.update({'operating_definition':{'flow_L_h':F,'inhibition':not a.disable_inhibition,'complete_ethanol_balance':a.complete_ethanol_balance,'horizon_h':10,'Ki_g_L':KI,'death_rate':0,'feed_glucose_g_L':[20]if a.case=='shared'else[10,10]},'source_files':[{'path':str(x.relative_to(ROOT)),'sha256':sha(x)}for x in SOURCES[a.case].iterdir()if x.suffix in ['.m','.mat']], 'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},'source_interpretation':'Selected07 source variant; no source death term; exact plot matching cannot distinguish07 vs08 code variants.','scope':'Consistent direct KKT calculation of stated source bounds, stopped before their exact LP feasibility boundary when necessary. No quiescence or maintenance relaxation.','script_sha256':sha(Path(__file__))})
 t,y,v,lev=data
 (dest/'audit.json').write_text(json.dumps(r,indent=2));(dest/'run_script.py').write_text(Path(__file__).read_text())
 np.savez_compressed(dest/'trajectory.npz',time_h=t,amounts=y,flux=v,secretion_level=lev)
 np.savetxt(dest/'trajectory.csv',np.c_[t,V0+F*t,y[:,:5]/(V0+F*t[:,None]),v[:,:,12],v[:,:,27],v[:,:,23]],delimiter=',',header='t_h,V_L,X1_gDW_L,X2_gDW_L,G1_or_shared_g_L,G2_g_L,E_g_L,vbio1,vbio2,vG1,vG2,vE1,vE2',comments='')
 print(json.dumps({k:r[k]for k in ['case','full_10h','terminal_time_h','terminal_concentrations','max_stored_LP_growth_gap','tracked_balance_errors_g','elapsed_s']},indent=2))
if __name__=='__main__':main()
