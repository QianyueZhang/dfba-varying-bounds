"""Repeated-light CSTR loading study: source kinetics and reactor retained.

All outputs stay beside this script. No canonical thesis/result files are changed.
Commands: preflight, run --case light|dark [--label ...], figures, audit.
"""
from pathlib import Path
import sys, os, json, time, hashlib, argparse, platform
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research_completion/scripts'))
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
from cstr_experiments import CSTRModel
from interior_solver import solve
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog
from scipy.integrate import solve_ivp
import scipy

VOLUME=125.; FLOW=1.; D=FLOW/VOLUME
KLA=.25
O_SAT=1000*55.56/(25131+709.2*25)*.21
C_SAT=1000*np.exp(-8.1403+842.9/(25+151.5))*.00032
INLET=np.array([10.,17.5,O_SAT,C_SAT,0.])
Y0=np.r_[VOLUME,.025,.025,INLET]
HORIZON=72.
AREA=.25
IRRADIANCE=60.
STATE_NAMES=['V_L','X_SC_gDW_L','X_RC_gDW_L','G_mmol_L','Ac_mmol_L','O2_mmol_L','CO2_mmol_L','E_mmol_L']
SOLUTES=['glucose','acetate','oxygen','carbon_dioxide','ethanol']


def light_fraction(t):
 return max(0.,(np.sin(np.pi*t/24)**2-np.sin(5*np.pi/24)**2)/(1-np.sin(5*np.pi/24)**2))


class Model(CSTRModel):
 def __init__(self,name,epsilon=1e-8,light=True):
  super().__init__(name)
  self.epsilon=epsilon;self.light=light
  path=ROOT/'data/models/cstr'/('iND750.mat' if name=='SC' else 'iRC1080.mat')
  raw=next(v for k,v in loadmat(path,simplify_cells=True).items() if not k.startswith('__'))
  self.raw=raw;self.ids={str(r):i for i,r in enumerate(raw['rxns'])}
  self.l=np.maximum(np.asarray(raw['lb'],float).ravel()-epsilon,-1000.)
  self.u=np.minimum(np.asarray(raw['ub'],float).ravel()+epsilon,1000.)
  if name=='RC':
   starch=self.ids['SK_starch300_h'];self.l[starch]=-epsilon;self.u[starch]=epsilon
 def bounds(self,state):
  t,y=state;l,u=super().bounds(state)
  if self.name=='RC':
   photon_capacity=(IRRADIANCE*AREA*3.6/(y[0]*max(y[2],1e-12)))*light_fraction(t) if self.light else 0.
   u[self.photon[0]]=photon_capacity+self.epsilon
  return l,u
 def lp(self,state):
  l,u=self.bounds(state)
  result=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=np.c_[l,u],method='highs',
                options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
  if not result.success:
   # HiGHS presolve can classify 2*epsilon-wide fixed bounds as inconsistent
   # at epsilon near feasibility tolerance; re-solve the identical LP directly.
   result=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=np.c_[l,u],method='highs',
                 options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'presolve':False})
  if not result.success:raise RuntimeError((self.name,state,result.message))
  return result


def summary_flux(model,flux):
 r={key:float(flux[index]) for key,index in model.exchange.items()}
 r['growth_h_inv']=float(flux[model.bio])
 if model.name=='RC':
  r['net_oxygen']=r['oxygen']+r['photosynthetic_oxygen']
  r['solar_photon']=float(flux[model.photon[0]])
 return r


def protocol():
 return {'initial_state':dict(zip(STATE_NAMES,Y0.tolist())), 'inlet_mmol_L':dict(zip(SOLUTES,INLET.tolist())),
         'horizon_h':HORIZON,'flow_L_h':FLOW,'dilution_h_inv':D,'temperature_C':25,'gas_kLa_h_inv':KLA,
         'air_equilibrium_mmol_L':{'oxygen':O_SAT,'carbon_dioxide':C_SAT},
         'atmosphere_atm':{'oxygen':.21,'carbon_dioxide':.00032},
         'illuminated_area_m2':AREA,'peak_irradiance_umol_m2_s':IRRADIANCE,
         'solar_capacity':'I_peak * light_fraction(t) * area * 3.6 / (V * X_RC), mmol photons/gDW/h',
         'light_fraction':'max(0,(sin(pi*t/24)^2-sin(5*pi/24)^2)/(1-sin(5*pi/24)^2))',
         'light_scope':'Incident photons form an upper capacity. Full incident availability is assumed; optical attenuation is not resolved.',
         'maintenance':'Raw ATPM unchanged: SC lower0; RC fixed0.183 mmol/gDW/h (before numerical bound relaxation).',
         'nutrients':'Other exchange bounds retained from reconstructions; untracked mineral nutrients are implicit reservoirs. No full-medium calibration.',
         'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
         'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OUT/'interior_solver.py',ROOT/'research_completion/scripts/cstr_experiments.py',ROOT/'data/models/cstr/iND750.mat',ROOT/'data/models/cstr/iRC1080.mat']}}


def preflight():
 records=[]
 for name,light,t in [('SC',True,0.),('RC',True,0.),('RC',True,12.)]:
    for epsilon in [1e-8,1e-9,0.]:
     m=Model(name,epsilon=epsilon,light=light);lp=m.lp((t,Y0))
     rec={'organism':name,'light':light,'t_h':t,'epsilon':epsilon,'LP':summary_flux(m,lp.x),'barrier':[]}
     for mu in [1e-8,1e-10,1e-12] if epsilon else []:
      try:
       s=solve(m,(t,Y0),mu=mu,tol=1e-12,maxiter=180)
       rec['barrier'].append({'mu':mu,'success':True,**summary_flux(m,s[0]),'diagnostics':s[-1],
                              'LP_growth_gap':float(lp.x[m.bio]-s[0][m.bio])})
      except RuntimeError as exc:rec['barrier'].append({'mu':mu,'success':False,'error':str(exc)})
     records.append(rec)
     print(name,light,t,epsilon,'LP',lp.x[m.bio],[(r['mu'],r.get('growth_h_inv'),r['success']) for r in rec['barrier']],flush=True)
     (OUT/'preflight.json').write_text(json.dumps({'protocol':protocol(),'checks':records},indent=2)+'\n')
 (OUT/'preflight.json').write_text(json.dumps({'protocol':protocol(),'checks':records},indent=2)+'\n')


def simulate(case,label,mu,epsilon,rtol,atol,max_step):
 models=[Model(n,epsilon,case=='light') for n in ['SC','RC']]
 warm=[None,None];n_rhs=0;max_eq=max_pd=max_relative_comp=0.;minimum_slack=1.;minimum_raw_state=Y0.copy();fallbacks=0
 flux_times=[];fluxes=[[],[]];lowers=[[],[]];uppers=[[],[]];sample_diagnostics=[]
 def intracellular(t,y):
  nonlocal max_eq,max_pd,max_relative_comp,minimum_slack,fallbacks
  values=[]
  for i,m in enumerate(models):
   try:s=solve(m,(t,y[:8]),mu=mu,tol=1e-10,warm=warm[i],maxiter=180)
   except RuntimeError:
    fallbacks+=1;s=solve(m,(t,y[:8]),mu=mu,tol=1e-10,maxiter=180)
   warm[i]=s[:4];values.append(s[0]);diag=s[-1]
   max_eq=max(max_eq,diag['equality_residual']);max_pd=max(max_pd,diag['primal_dual_residual']);minimum_slack=min(minimum_slack,diag['minimum_slack'])
   max_relative_comp=max(max_relative_comp,diag['complementarity_relative_residual'])
  return values
 def rhs(t,y):
  nonlocal n_rhs,minimum_raw_state
  n_rhs+=1;minimum_raw_state=np.minimum(minimum_raw_state,y[:8]);v,w=intracellular(t,y)
  V,x1,x2,G,Ac,O2,CO2,E=y[:8];sc,rc=(m.exchange for m in models)
  cell=np.array([v[sc['glucose']]*x1,
      v[sc['acetate']]*x1+w[rc['acetate']]*x2,
      v[sc['oxygen']]*x1+(w[rc['oxygen']]+w[rc['photosynthetic_oxygen']])*x2,
      v[sc['carbon_dioxide']]*x1+w[rc['carbon_dioxide']]*x2,
      v[sc['ethanol']]*x1+w[rc['ethanol']]*x2])
  gas=np.array([KLA*(O_SAT-O2),KLA*(C_SAT-CO2)])
  dilute=FLOW/V*(INLET-y[3:8]);dy=dilute+cell;dy[2:4]+=gas
  # Integrated effluent, cell exchange and gas exchange provide independent
  # named balances; these accumulators do not feed back into physical states.
  result=np.r_[0.,(v[models[0].bio]-FLOW/V)*x1,(w[models[1].bio]-FLOW/V)*x2,dy,
               FLOW/V*y[3:8],cell,gas]
  if n_rhs%100==0:print(label,'rhs',n_rhs,'t',t,'state',y[1:8],flush=True)
  return result
 started=time.perf_counter();state=np.r_[Y0,np.zeros(12)];times=[];states=[];segments=[]
 boundaries=sorted(set([0.,HORIZON]+[24.*day+phase for day in range(int(np.ceil(HORIZON/24))) for phase in [5.,12.,19.,24.] if 0.<24.*day+phase<HORIZON]))
 for a,b in zip(boundaries[:-1],boundaries[1:]):
  sol=solve_ivp(rhs,(a,b),state,method='BDF',rtol=rtol,atol=atol,max_step=max_step,dense_output=True)
  if not sol.success:raise RuntimeError(sol.message)
  tt=np.linspace(a,b,int(round((b-a)/.1))+1)
  if times:tt=tt[1:]
  times.extend(tt);states.extend(sol.sol(tt).T);state=sol.y[:,-1]
  segments.append({'start':a,'end':b,'nfev':sol.nfev,'njev':sol.njev,'nlu':sol.nlu,'endpoint':state[:8].tolist()})
  print(label,'segment',b,state[:8],flush=True)
 elapsed=time.perf_counter()-started;tt=np.array(times);yy=np.array(states)
 for t,y in zip(tt,yy):
  vv=intracellular(t,y)
  for i,m in enumerate(models):
   l,u=m.bounds((t,y[:8]));fluxes[i].append(vv[i]);lowers[i].append(l);uppers[i].append(u)
  if any(np.isclose(t,k,atol=1e-9) for k in boundaries):
   for m,v in zip(models,vv):
    lp=m.lp((t,y[:8]));sample_diagnostics.append({'time_h':float(t),'organism':m.name,
       'barrier':summary_flux(m,v),'LP':summary_flux(m,lp.x),'LP_growth_gap':float(lp.x[m.bio]-v[m.bio]),
       'LP_equality_residual':float(np.max(np.abs(m.A@lp.x-m.b)))})
 np.savez_compressed(OUT/f'{label}.npz',time_h=tt,states=yy[:,:8],integrals=yy[:,8:],
     sc_flux=np.array(fluxes[0]),rc_flux=np.array(fluxes[1]),sc_lower=np.array(lowers[0]),sc_upper=np.array(uppers[0]),
     rc_lower=np.array(lowers[1]),rc_upper=np.array(uppers[1]))
 np.savetxt(OUT/f'{label}_trajectory.csv',np.c_[tt,yy[:,:8]],delimiter=',',header='t_h,'+','.join(STATE_NAMES),comments='')
 balance=state[3:8]-Y0[3:8]-(D*INLET*HORIZON-state[8:13]+state[13:18]+np.r_[0.,0.,state[18:20],0.])
 result={'case':case,'label':label,'protocol':protocol(),'numerics':{'mu':mu,'epsilon':epsilon,'rtol':rtol,'atol':atol,'max_step':max_step,'feasibility_stationarity_tolerance':1e-10,'relative_complementarity_tolerance':2e-8,'solver':'row-scaled equalities; componentwise complementarity checked relative to mu with floating-point bound-subtraction allowance'},
   'terminal':dict(zip(STATE_NAMES,state[:8].tolist())),'minimum_states':dict(zip(STATE_NAMES,yy[:,:8].min(axis=0).tolist())),
   'maximum_states':dict(zip(STATE_NAMES,yy[:,:8].max(axis=0).tolist())),
   'rhs_calls':n_rhs,'elapsed_integration_seconds':elapsed,'cold_restart_count':fallbacks,'segments':segments,
   'max_equality_residual':max_eq,'max_primal_dual_residual':max_pd,'max_effective_relative_complementarity_residual':max_relative_comp,'minimum_slack':minimum_slack,
   'minimum_raw_rhs_states':minimum_raw_state.tolist(),'balance_residual_mmol_L':dict(zip(SOLUTES,balance.tolist())),
   'integrated_effluent':dict(zip(SOLUTES,state[8:13].tolist())),
   'integrated_cell_net_secretion':dict(zip(SOLUTES,state[13:18].tolist())),
   'integrated_gas_net_inward':dict(zip(['oxygen','carbon_dioxide'],state[18:20].tolist())),
   'sampled_LP_checks':sample_diagnostics}
 (OUT/f'{label}.json').write_text(json.dumps(result,indent=2)+'\n')
 print(label,'DONE',json.dumps({k:result[k] for k in ['terminal','max_equality_residual','max_primal_dual_residual','balance_residual_mmol_L']}),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','run']);parser.add_argument('--case',choices=['light','dark'],default='light');parser.add_argument('--label');parser.add_argument('--mu',type=float,default=1e-10);parser.add_argument('--epsilon',type=float,default=1e-8);parser.add_argument('--rtol',type=float,default=1e-7);parser.add_argument('--atol',type=float,default=1e-10);parser.add_argument('--max-step',type=float,default=.2);parser.add_argument('--initial-biomass',type=float,default=.025);parser.add_argument('--horizon',type=float,default=72.)
 args=parser.parse_args()
 if args.initial_biomass<=0 or args.horizon<=0:raise ValueError('Positive biomass and horizon required')
 Y0[1:3]=args.initial_biomass;HORIZON=args.horizon
 if args.command=='preflight':preflight()
 else:simulate(args.case,args.label or args.case,args.mu,args.epsilon,args.rtol,args.atol,args.max_step)
