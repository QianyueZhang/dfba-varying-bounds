"""Consistent finite-barrier iJR904 sequential-substrate benchmark.

Source kinetics and mass units are retained; the active-growth calculation
ends before the fixed ATP maintenance demand exhausts the feasible medium.
No post-depletion growth or negative substrate continuation is imposed.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from scipy.optimize import linprog
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research_completion/runs/diauxic_consistent_2026-10-21'
sys.path.insert(0,str(ROOT/'research_completion/runs/cstr_supported_2026-10-21'))
from interior_solver import solve

def source_provenance(model):
 paths=[model.source/'iJR904.mat',model.source/'E-coli_Model.mat',Path(__file__)]
 historical=ROOT/'data/diauxic_implementation_provenance.json'
 return {'source_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
         'historical_implementation':json.loads(historical.read_text())}

class Model:
 name='iJR904 sequential substrate';bio=268;g=397;xy=741;o=499
 def __init__(self,epsilon=1e-7):
  self.epsilon=epsilon
  self.source=ROOT/'data/models/diauxic'
  raw=loadmat(self.source/'iJR904.mat',simplify_cells=True)['iJR904']
  processed=loadmat(self.source/'E-coli_Model.mat',simplify_cells=True)['Struct']
  self.A=csc_matrix(processed['A']);self.b=np.asarray(processed['b'],float).ravel();self.c=np.asarray(processed['c'],float).ravel()
  self.l=np.maximum(np.asarray(raw['lb'],float),-1000)-epsilon
  self.u=np.minimum(np.asarray(raw['ub'],float),1000)+epsilon
  self.u[[self.g,self.xy,self.o]]=epsilon
  self.raw=raw
 def bounds(self,state):
  G,Z=np.maximum(np.asarray(state)[1:3],0)
  q=np.array([10*G/(1+G),8*Z/(1+Z)/(1+G/.1),20/1.1])
  l=self.l.copy();u=self.u.copy();l[[self.g,self.xy,self.o]]=-q
  return l,u
 def lp(self,state):
  l,u=self.bounds(state)
  r=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=np.c_[l,u],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  if not r.success:raise RuntimeError((state,r.message))
  return r

def integrate(label,rtol=1e-7,atol=1e-9,mu=1e-7,epsilon=1e-7):
 model=Model(epsilon);warm=None;calls=0;last=-1;res=[]
 def flux(y):
  nonlocal warm,calls,last
  try:s=solve(model,y,mu=mu,tol=1e-10,complementarity_rtol=1e-7,warm=warm)
  except RuntimeError:
   print(label,'cold-start retry',y,flush=True)
   s=solve(model,y,mu=mu,tol=1e-10,complementarity_rtol=1e-7,warm=None)
  warm=s[:4];calls+=1
  if calls%100==0:print(label,calls,'state',y[:3],flush=True)
  return s[0],s[-1]
 def rhs(t,y):
  v,d=flux(y);res.append(d)
  return np.r_[v[model.bio]*y[0],.180156*v[model.g]*y[0],.15013*v[model.xy]*y[0]]
 def end(t,y):
  return y[2]-.08
 end.terminal=True;end.direction=-1
 begin=time.perf_counter()
 sol=solve_ivp(rhs,(0,12),[.01,15,8],method='BDF',rtol=rtol,atol=atol,max_step=.01,events=end,dense_output=True)
 if not sol.success:raise RuntimeError(sol.message)
 t=np.linspace(0,sol.t[-1],501);y=sol.sol(t).T;vv=[];dd=[];lb=[];ub=[];warm=None
 np.savez_compressed(OUT/(label+'_integration.npz'),t=t,y=y,solver_t=sol.t,solver_y=sol.y)
 for yy in y:
  v,d=flux(yy);vv.append(v[[model.bio,model.g,model.xy,model.o]]);dd.append(d)
  l,u=model.bounds(yy);lb.append(l[[model.g,model.xy,model.o]]);ub.append(u[[model.g,model.xy,model.o]])
 np.savez_compressed(OUT/(label+'.npz'),t=t,y=y,flux=np.array(vv),lower=np.array(lb),upper=np.array(ub),solver_t=sol.t,solver_y=sol.y)
 record={'label':label,'success':bool(sol.success),'end_h':float(sol.t[-1]),'xylose_stop_g_L':.08,'initial':[.01,15,8],'terminal':y[-1].tolist(),'mu':mu,'epsilon':epsilon,'rtol':rtol,'atol':atol,'calls':calls,'seconds':time.perf_counter()-begin,'shape':list(model.A.shape),'maximum_residuals':{key:max(d[key] for d in dd+res) for key in ['equality_residual','primal_dual_residual','complementarity_relative_residual']},'minimum_slack':min(d['minimum_slack'] for d in dd+res),'minimum_concentrations':y.min(0).tolist(),**source_provenance(model),'method':'Direct finite-barrier intracellular solution with BDF extracellular integration; fixed ATPM7.6, stopping at1% residual initial xylose. No stationary-phase extension.'}
 (OUT/(label+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
 return record

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--label',default='primary');p.add_argument('--refine',action='store_true');p.add_argument('--mu',type=float,default=1e-7);p.add_argument('--epsilon',type=float,default=1e-7);a=p.parse_args()
 integrate(a.label,rtol=2e-8 if a.refine else 1e-7,atol=2e-10 if a.refine else 1e-9,mu=a.mu,epsilon=a.epsilon)
