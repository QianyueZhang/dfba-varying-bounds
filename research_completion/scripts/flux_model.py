"""Genome-scale varying-bound models for microbial feed control.
The non-interacting control benchmark imposes no forced ethanol production
and solves a consistent finite-barrier intracellular state.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix, bmat, diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[2]
class FluxModel:
 def __init__(self,name,epsilon=1e-3):
  self.name=name;self.epsilon=epsilon
  p=ROOT/'data/models/yeast_control'/f'{name}_Model.mat'
  m=loadmat(p,simplify_cells=True)['Struct'];self.A=csc_matrix(m['A']);self.b=np.asarray(m['b'],float).ravel();self.c=np.asarray(m['c'],float).ravel()
  rawname={'SC':'SCerevisiae_Data.mat','KP':'KPhaffii_Data.mat'}[name]
  raw=next(v for k,v in loadmat(p.parent/rawname,simplify_cells=True).items() if not k.startswith('__'))
  self.l=np.asarray(raw['lb'],float).ravel()-epsilon;self.u=np.asarray(raw['ub'],float).ravel()+epsilon
  self.l[np.isneginf(self.l)]=-1000;self.u[np.isposinf(self.u)]=1000
  self.g,self.bio,self.qmax,self.ks=(2664,2984,12.9,6.95) if name=='SC' else (113,88,6.,.385)
  self.u[self.g]=epsilon
  self.provenance={'processed_matrix':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw_bounds_source':str((p.parent/rawname).relative_to(ROOT)),'shape':list(self.A.shape),'epsilon':epsilon,'glucose_index_1based':self.g+1,'biomass_index_1based':self.bio+1,'ethanol_policy':'No forced production for non-interacting control benchmark'}
 def bounds(self,S):
  l=self.l.copy();u=self.u.copy();l[self.g]=-self.qmax*max(S,0)/(self.ks+max(S,0))-self.epsilon
  return l,u
 def lp(self,S):
  l,u=self.bounds(S)
  r=linprog(-self.c,A_eq=self.A,b_eq=self.b,bounds=np.c_[l,u],method='highs',options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
  if not r.success:raise RuntimeError((self.name,S,r.message))
  return r
 def barrier(self,S,mu=1e-6,warm=None,tol=1e-7,maxiter=160):
  l,u=self.bounds(S);n=len(l);m=len(self.b)
  if warm is None:
   lp=self.lp(S);x=np.clip(lp.x,l+np.minimum(.01,(u-l)*.1),u-np.minimum(.01,(u-l)*.1));lam=-lp.eqlin.marginals
   mus=[1e-2,1e-3,1e-4,1e-5,mu] if mu<1e-5 else [mu]
  else:
   x,lam=(a.copy() for a in warm);x=np.clip(x,l+np.minimum(1e-7,(u-l)*.001),u-np.minimum(1e-7,(u-l)*.001));mus=[mu]
  iterations=0
  for muv in mus:
   for it in range(maxiter):
    sl=x-l;su=u-x;rd=-self.c+self.A.T@lam-muv/sl+muv/su;rp=self.A@x-self.b
    norm=max(np.max(np.abs(rd)),np.max(np.abs(rp)))
    if norm<tol:break
    D=muv/sl**2+muv/su**2
    K=bmat([[diags(D),self.A.T],[self.A,None]],format='csc')
    d=spsolve(K,-np.r_[rd,rp]);dx=d[:n];dl=d[n:]
    alpha=1.
    if np.any(dx<0):alpha=min(alpha,.995*np.min(-sl[dx<0]/dx[dx<0]))
    if np.any(dx>0):alpha=min(alpha,.995*np.min(su[dx>0]/dx[dx>0]))
    # Residual norm line search for infeasible-start Newton.
    old=np.linalg.norm(np.r_[rd,rp]);accepted=False
    for _ in range(35):
     xn=x+alpha*dx;ln=lam+alpha*dl
     rn=-self.c+self.A.T@ln-muv/(xn-l)+muv/(u-xn)
     merit=np.linalg.norm(np.r_[rn,self.A@xn-self.b])
     if np.isfinite(merit) and merit<=(1-1e-4*alpha)*old:
      accepted=True;break
     alpha*=.5
    if not accepted:raise RuntimeError(f'{self.name}: Newton line search failed at S={S}, mu={muv}, residual={norm}')
    x=xn;lam=ln;iterations+=1
   else:raise RuntimeError(f'{self.name}: Newton did not converge at S={S}, mu={muv}, residual={norm}')
  return x,lam,{'equality_residual':float(np.max(np.abs(self.A@x-self.b))),'stationarity_residual':float(np.max(np.abs(-self.c+self.A.T@lam-mu/(x-l)+mu/(u-x)))),'minimum_slack':float(min(np.min(x-l),np.min(u-x))),'newton_steps':iterations,'growth':float(x[self.bio]),'glucose_flux':float(x[self.g])}
if __name__=='__main__':
 out=[]
 for name in ['SC','KP']:
  mod=FluxModel(name);t=time.perf_counter();lp=mod.lp(20);print(name,'LP',lp.fun,lp.x[mod.g],'seconds',time.perf_counter()-t,flush=True)
  t=time.perf_counter();x,la,d=mod.barrier(20);print(name,'barrier',d,'seconds',time.perf_counter()-t,flush=True);out.append(mod.provenance|d|{'lp_growth':-lp.fun})
 (ROOT/'research_completion/runs/barrier_initial_check.json').write_text(json.dumps(out,indent=2))
